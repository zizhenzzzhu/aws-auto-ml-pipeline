"""
Sanitized data pull stage for an AWS auto-ML pipeline.

This module keeps connection details and SQL outside of source control. It can
pull data with pandas for small checks or with PySpark for production-scale
training data preparation, then write Parquet that PyTorch training jobs can
consume.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable


SECRET_KEY_ALIASES = {
    "server_hostname": ("server_hostname", "DATABRICKS_SERVER_HOSTNAME", "DATA_PLATFORM_HOST"),
    "http_path": ("http_path", "DATABRICKS_HTTP_PATH", "DATA_PLATFORM_HTTP_PATH"),
    "access_token": ("access_token", "token", "DATABRICKS_TOKEN", "DATA_PLATFORM_TOKEN"),
}


@dataclass
class DataPullConfig:
    source_type: str = field(default_factory=lambda: os.getenv("DATA_SOURCE_TYPE", "databricks_sql"))
    mode: str = "cloud"
    catalog: str = field(default_factory=lambda: os.getenv("DATA_PLATFORM_CATALOG", ""))
    schema: str = field(default_factory=lambda: os.getenv("DATA_PLATFORM_SCHEMA", ""))
    secret_name: str | None = field(default_factory=lambda: os.getenv("DATA_PLATFORM_SECRET_NAME"))
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-west-2"))
    server_hostname: str | None = field(default_factory=lambda: os.getenv("DATABRICKS_SERVER_HOSTNAME"))
    http_path: str | None = field(default_factory=lambda: os.getenv("DATABRICKS_HTTP_PATH"))
    access_token: str | None = field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN"))
    port: int = field(default_factory=lambda: int(os.getenv("DATA_PLATFORM_PORT", "443")))
    jdbc_driver: str = field(
        default_factory=lambda: os.getenv("DATA_PLATFORM_JDBC_DRIVER", "com.databricks.client.jdbc.Driver")
    )
    s3_input_uri: str | None = field(default_factory=lambda: os.getenv("S3_INPUT_URI"))
    csv_header: bool = field(default_factory=lambda: os.getenv("CSV_HEADER", "true").lower() == "true")
    csv_infer_schema: bool = field(default_factory=lambda: os.getenv("CSV_INFER_SCHEMA", "true").lower() == "true")


class DataPullStage:
    """Runs sanitized SQL pulls from a data platform."""

    def __init__(self, config: DataPullConfig | None = None):
        self.config = config or DataPullConfig()

    def run_pandas(self, query: str | None = None, sql_file: str | Path | None = None):
        """Run a query into a pandas DataFrame for lightweight validation."""
        import pandas as pd

        sql = self._resolve_query(query=query, sql_file=sql_file)
        if self.config.source_type == "s3_csv":
            return self._read_csv_pandas()

        from databricks import sql as databricks_sql

        creds = self._databricks_credentials()

        with databricks_sql.connect(
            server_hostname=creds["server_hostname"],
            http_path=creds["http_path"],
            access_token=creds["access_token"],
            catalog=self.config.catalog or None,
            schema=self.config.schema or None,
        ) as conn:
            return pd.read_sql(sql, conn)

    def run_spark(self, query: str | None = None, sql_file: str | Path | None = None, spark=None):
        """Run a query into a PySpark DataFrame through JDBC."""
        spark = spark or create_spark_session()

        if self.config.source_type == "s3_csv":
            return (
                spark.read.option("header", str(self.config.csv_header).lower())
                .option("inferSchema", str(self.config.csv_infer_schema).lower())
                .csv(self._required_s3_input_uri())
            )

        if self.config.source_type == "snowflake_sql":
            sql = self._resolve_query(query=query, sql_file=sql_file)
            raise NotImplementedError(
                "snowflake_sql Spark mode is not yet implemented. "
                "Use --engine pandas (run_local_pipeline.py) for local runs, "
                "or wire a Snowflake JDBC connector in create_spark_session()."
            )

        if self.config.source_type != "databricks_sql":
            raise ValueError(f"Unsupported source_type: {self.config.source_type}")

        sql = self._resolve_query(query=query, sql_file=sql_file)

        creds = self._databricks_credentials()
        jdbc_url = self._jdbc_url(creds["server_hostname"], creds["http_path"])

        return (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("driver", self.config.jdbc_driver)
            .option("query", sql)
            .option("user", "token")
            .option("password", creds["access_token"])
            .load()
        )

    def write_training_parquet(
        self,
        output_uri: str,
        query: str | None = None,
        sql_file: str | Path | None = None,
        partition_columns: Iterable[str] | None = None,
        mode: str = "overwrite",
        spark=None,
    ) -> str:
        """Pull training data with Spark and write Parquet to local storage or S3."""
        frame = self.run_spark(query=query, sql_file=sql_file, spark=spark)

        if partition_columns:
            writer = frame.write.mode(mode).partitionBy(*partition_columns)
        else:
            writer = frame.write.mode(mode)

        writer.parquet(output_uri)
        return output_uri

    def _cloud_credentials(self) -> dict[str, str]:
        if not self.config.secret_name:
            raise ValueError("DATA_PLATFORM_SECRET_NAME is required for cloud mode.")

        import boto3

        client = boto3.client("secretsmanager", region_name=self.config.region)
        response = client.get_secret_value(SecretId=self.config.secret_name)
        secret = json.loads(response["SecretString"])

        return {
            "server_hostname": self._lookup_secret(secret, "server_hostname"),
            "http_path": self._lookup_secret(secret, "http_path"),
            "access_token": self._lookup_secret(secret, "access_token"),
        }

    def _databricks_credentials(self) -> dict[str, str]:
        if self.config.mode == "cloud":
            return self._cloud_credentials()

        if self.config.mode not in {"basic", "local"}:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        if not self.config.server_hostname or not self.config.http_path or not self.config.access_token:
            raise ValueError(
                "DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN are required."
            )

        return {
            "server_hostname": self.config.server_hostname,
            "http_path": self.config.http_path,
            "access_token": self.config.access_token,
        }

    def _jdbc_url(self, server_hostname: str, http_path: str) -> str:
        schema_path = self.config.schema or "default"
        catalog_param = f";ConnCatalog={self.config.catalog}" if self.config.catalog else ""
        schema_param = f";ConnSchema={self.config.schema}" if self.config.schema else ""
        return (
            f"jdbc:databricks://{server_hostname}:{self.config.port}/{schema_path};"
            f"transportMode=http;ssl=1;httpPath={http_path};AuthMech=3"
            f"{catalog_param}{schema_param}"
        )

    def _required_s3_input_uri(self) -> str:
        if not self.config.s3_input_uri:
            raise ValueError("S3_INPUT_URI or --s3-input-uri is required when source_type=s3_csv.")
        return self.config.s3_input_uri

    def _read_csv_pandas(self):
        import pandas as pd

        input_uri = self._required_s3_input_uri()
        if not input_uri.startswith("s3://"):
            return pd.read_csv(input_uri, header=0 if self.config.csv_header else None)

        import boto3

        parsed = urlparse(input_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key or key.endswith("/"):
            raise ValueError("Pandas S3 CSV mode requires a single s3://bucket/key.csv object, not a prefix.")

        response = boto3.client("s3", region_name=self.config.region).get_object(Bucket=bucket, Key=key)
        return pd.read_csv(response["Body"], header=0 if self.config.csv_header else None)

    @staticmethod
    def _resolve_query(query: str | None = None, sql_file: str | Path | None = None) -> str:
        if query and sql_file:
            raise ValueError("Provide either query or sql_file, not both.")
        if sql_file:
            return Path(sql_file).read_text(encoding="utf-8")
        if query:
            return query
        raise ValueError("Provide query or sql_file.")

    @staticmethod
    def _lookup_secret(secret: dict[str, object], logical_key: str) -> str:
        for alias in SECRET_KEY_ALIASES[logical_key]:
            value = secret.get(alias)
            if value is not None:
                return str(value)
        raise KeyError(f"Secret is missing a value for {logical_key}.")


def create_spark_session(app_name: str = "aws-auto-ml-data-pull", extra_conf: dict[str, str] | None = None):
    """Create a Spark session, automatically configuring S3A when running locally.

    Set SPARK_JARS_PACKAGES to include hadoop-aws + aws-java-sdk-bundle for local S3 access, e.g.:
      org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.367
    """
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)

    jars_packages = os.getenv("SPARK_JARS_PACKAGES")
    if jars_packages:
        builder = builder.config("spark.jars.packages", jars_packages)

    # S3A filesystem — applied when running outside EMR/Databricks where these are pre-configured.
    s3a_settings: dict[str, str] = {}
    if os.getenv("SPARK_HADOOP_FS_S3A_IMPL"):
        s3a_settings["spark.hadoop.fs.s3a.impl"] = os.environ["SPARK_HADOOP_FS_S3A_IMPL"]
    if os.getenv("AWS_ACCESS_KEY_ID"):
        s3a_settings["spark.hadoop.fs.s3a.access.key"] = os.environ["AWS_ACCESS_KEY_ID"]
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        s3a_settings["spark.hadoop.fs.s3a.secret.key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if os.getenv("AWS_SESSION_TOKEN"):
        s3a_settings["spark.hadoop.fs.s3a.session.token"] = os.environ["AWS_SESSION_TOKEN"]

    for key, value in s3a_settings.items():
        builder = builder.config(key, value)

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


class ParquetTorchDataset:
    """
    Minimal iterable PyTorch dataset over Parquet output.

    In production, use this from the training container after Spark has written
    Parquet to the mounted training channel or an accessible object-store path.
    """

    def __init__(self, parquet_uri: str, feature_columns: list[str], label_column: str, batch_size: int = 1024):
        self.parquet_uri = parquet_uri
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.batch_size = batch_size

    def __iter__(self):
        import pyarrow.dataset as ds
        import torch

        dataset = ds.dataset(self.parquet_uri, format="parquet")
        columns = self.feature_columns + [self.label_column]

        for batch in dataset.to_batches(columns=columns, batch_size=self.batch_size):
            frame = batch.to_pandas()
            features = torch.tensor(frame[self.feature_columns].values, dtype=torch.float32)
            labels = torch.tensor(frame[self.label_column].values)
            yield features, labels

    def to_iterable_dataset(self):
        """Return a torch IterableDataset wrapper for DataLoader integration."""
        import torch

        source = self

        class _ParquetIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter(source)

        return _ParquetIterableDataset()


def create_torch_dataloader(
    parquet_uri: str,
    feature_columns: list[str],
    label_column: str,
    batch_size: int = 1024,
):
    """Create a PyTorch DataLoader over Parquet produced by the Spark stage."""
    import torch

    dataset = ParquetTorchDataset(
        parquet_uri=parquet_uri,
        feature_columns=feature_columns,
        label_column=label_column,
        batch_size=batch_size,
    ).to_iterable_dataset()
    return torch.utils.data.DataLoader(dataset, batch_size=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull sanitized training data for the AWS auto-ML pipeline.")
    parser.add_argument("--source-type", default=os.getenv("DATA_SOURCE_TYPE", "databricks_sql"), choices=("databricks_sql", "s3_csv", "snowflake_sql"))
    parser.add_argument("--mode", default=os.getenv("DATA_PULL_MODE", "cloud"), choices=("cloud", "basic", "local"))
    parser.add_argument("--engine", default="spark", choices=("spark", "pandas"))
    parser.add_argument("--query-file", required=True, help="Path to a local SQL file. Keep SQL out of source control.")
    parser.add_argument("--output-uri", help="Local path or S3 URI for pulled data.")
    parser.add_argument("--catalog", default=os.getenv("DATA_PLATFORM_CATALOG", ""))
    parser.add_argument("--schema", default=os.getenv("DATA_PLATFORM_SCHEMA", ""))
    parser.add_argument("--secret-name", default=os.getenv("DATA_PLATFORM_SECRET_NAME"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-west-2"))
    parser.add_argument("--partition-column", action="append", default=[])
    parser.add_argument("--s3-input-uri", default=os.getenv("S3_INPUT_URI"))
    parser.add_argument("--csv-header", default=os.getenv("CSV_HEADER", "true"))
    parser.add_argument("--csv-infer-schema", default=os.getenv("CSV_INFER_SCHEMA", "true"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataPullConfig(
        source_type=args.source_type,
        mode=args.mode,
        catalog=args.catalog,
        schema=args.schema,
        secret_name=args.secret_name,
        region=args.region,
        s3_input_uri=args.s3_input_uri,
        csv_header=str(args.csv_header).lower() == "true",
        csv_infer_schema=str(args.csv_infer_schema).lower() == "true",
    )
    stage = DataPullStage(config)

    if args.engine == "spark":
        if not args.output_uri:
            raise ValueError("--output-uri is required when --engine spark.")
        stage.write_training_parquet(
            sql_file=args.query_file,
            output_uri=args.output_uri,
            partition_columns=args.partition_column,
        )
        return

    frame = stage.run_pandas(sql_file=args.query_file)
    if args.output_uri:
        frame.to_parquet(args.output_uri, index=False)
    else:
        print(frame.head())


if __name__ == "__main__":
    main()
