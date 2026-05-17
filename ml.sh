#!/usr/bin/env bash
# ml.sh — Production entry point for the AWS Auto-ML Pipeline.
# Usage:
#   ./ml.sh                          # full run (ECR build + Airflow trigger)
#   ./ml.sh --local                  # run all stages locally (no Airflow)
#   ./ml.sh --dry-run                # validate + print commands, no AWS calls
#   ./ml.sh --skip-ecr               # skip Docker build/push
#   ./ml.sh --local --skip-data-pull # resume local run from ETL onwards
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/ml_config.yaml"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; NC=$'\033[0m'

ok()   { printf "  ${GREEN}✓${NC}  %s\n" "$1"; }
fail() { printf "  ${RED}✗${NC}  %s\n" "$1"; }
warn() { printf "  ${YELLOW}⚠${NC}  %s\n" "$1"; }
info() { printf "  ${DIM}→${NC}  %s\n" "$1"; }
hdr()  { printf "\n${BOLD}${BLUE}▶  %s${NC}\n" "$1"; }

yaml_get() {
  grep -E "^\s*${2}:" "${1}" 2>/dev/null | head -1 \
    | sed 's/.*: *//' | tr -d '"' | tr -d "'" | xargs 2>/dev/null || true
}

# ── Parse flags ───────────────────────────────────────────────────────────────
MODE="airflow"           # airflow | local
DRY_RUN=0
SKIP_ECR=0
FORCE_BUILD=0
SKIP_DATA_PULL=0
SKIP_AIRFLOW_REGISTER=0
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --local)                MODE="local" ;;
    --dry-run)              DRY_RUN=1;   PASSTHROUGH_ARGS+=("$arg") ;;
    --skip-ecr)             SKIP_ECR=1;  PASSTHROUGH_ARGS+=("$arg") ;;
    --force-build)          FORCE_BUILD=1; PASSTHROUGH_ARGS+=("$arg") ;;
    --skip-data-pull)       SKIP_DATA_PULL=1 ;;
    --skip-airflow-register) SKIP_AIRFLOW_REGISTER=1; PASSTHROUGH_ARGS+=("$arg") ;;
    --skip-trigger)         PASSTHROUGH_ARGS+=("$arg") ;;
    --config=*)             CONFIG_FILE="${arg#--config=}" ;;
    --help|-h)
      printf "Usage: ./ml.sh [--local] [--dry-run] [--skip-ecr] [--force-build]\n"
      printf "               [--skip-data-pull] [--skip-airflow-register] [--skip-trigger]\n"
      exit 0 ;;
    *) printf "${RED}Unknown flag: %s${NC}\n" "$arg"; exit 1 ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
printf "\n"
printf "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}${CYAN}║         AWS Auto-ML Pipeline  ·  ml.sh v3.0  (production)           ║${NC}\n"
if [[ "$MODE" == "local" ]]; then
printf "${BOLD}${CYAN}║         Mode: LOCAL  (run_local_pipeline.py)                        ║${NC}\n"
else
printf "${BOLD}${CYAN}║         Mode: AIRFLOW  (bootstrap → ECR → DAG trigger)              ║${NC}\n"
fi
[[ $DRY_RUN -eq 1 ]] && \
printf "${BOLD}${YELLOW}║         ⚠  DRY-RUN — no AWS calls will be made                      ║${NC}\n"
printf "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}\n"

# ── Step 1: Prerequisites ─────────────────────────────────────────────────────
hdr "Step 1/4  —  Prerequisites"

PREREQ_OK=1

check_cmd() {
  if command -v "$1" &>/dev/null; then
    ok "$1 found  ($(command -v "$1"))"
  else
    fail "$1 not found — $2"
    PREREQ_OK=0
  fi
}

check_cmd python3    "install Python 3.9+"
check_cmd aws        "install AWS CLI v2: https://aws.amazon.com/cli/"
if [[ "$MODE" == "airflow" && $SKIP_ECR -eq 0 ]]; then
  check_cmd docker   "install Docker Desktop to build/push the inference image"
fi
if [[ "$MODE" == "airflow" && $SKIP_AIRFLOW_REGISTER -eq 0 ]]; then
  check_cmd airflow  "install Apache Airflow or set airflow.provider=mwaa in ml_config.yaml"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  fail "Config file not found: $CONFIG_FILE"
  PREREQ_OK=0
else
  ok "Config file found: $CONFIG_FILE"
fi

if [[ $PREREQ_OK -eq 0 ]]; then
  printf "\n${RED}  Prerequisites failed — fix the items above and rerun.${NC}\n\n"
  exit 1
fi

# ── Step 2: Config validation ─────────────────────────────────────────────────
hdr "Step 2/4  —  Validating config"

info "Running bootstrap validation (python scripts/bootstrap_ml_pipeline.py --skip-ecr --skip-airflow-register --skip-trigger ${DRY_RUN:+--dry-run})…"
printf "\n"

python3 "${SCRIPT_DIR}/scripts/bootstrap_ml_pipeline.py" \
  --config "$CONFIG_FILE" \
  --skip-ecr \
  --skip-airflow-register \
  --skip-trigger \
  ${DRY_RUN:+--dry-run}

printf "\n"
ok "Config valid — all required fields present, S3 bucket reachable"

# ── Step 3: Pre-flight summary ────────────────────────────────────────────────
hdr "Step 3/4  —  Pre-flight summary"

PIPELINE=$(yaml_get "$CONFIG_FILE" "pipeline_name")
MODEL_GROUP=$(yaml_get "$CONFIG_FILE" "model_package_group")
ALGORITHM=$(yaml_get "$CONFIG_FILE" "algorithm")
LABEL_COL=$(yaml_get "$CONFIG_FILE" "label_column")
SOURCE_TYPE=$(yaml_get "$CONFIG_FILE" "source_type")
QUERY_FILE=$(yaml_get "$CONFIG_FILE" "query_file")
REGION=$(yaml_get "$CONFIG_FILE" "region")
DAG_ID=$(yaml_get "$CONFIG_FILE" "dag_id")
ECR_REPO=$(yaml_get "$CONFIG_FILE" "repository_name")
LR=$(yaml_get "$CONFIG_FILE" "learning_rate")
EPOCHS=$(yaml_get "$CONFIG_FILE" "epochs")

printf "\n"
printf "  ┌──────────────────────────────┬──────────────────────────────────────────┐\n"
printf "  │ %-28s │ %-40s │\n" "Pipeline"        "${PIPELINE:-—}"
printf "  │ %-28s │ %-40s │\n" "Mode"            "$MODE"
printf "  │ %-28s │ %-40s │\n" "Data source"     "${SOURCE_TYPE:-—}"
printf "  │ %-28s │ %-40s │\n" "Query file"      "${QUERY_FILE:-—}"
printf "  │ %-28s │ %-40s │\n" "Label column"    "${LABEL_COL:-—}"
printf "  │ %-28s │ %-40s │\n" "Algorithm"       "${ALGORITHM:-—}"
printf "  │ %-28s │ %-40s │\n" "Learning rate"   "${LR:-—}"
printf "  │ %-28s │ %-40s │\n" "Epochs"          "${EPOCHS:-—}"
printf "  │ %-28s │ %-40s │\n" "Model group"     "${MODEL_GROUP:-—}"
printf "  │ %-28s │ %-40s │\n" "AWS region"      "${REGION:-—}"
printf "  │ %-28s │ %-40s │\n" "Airflow DAG"     "${DAG_ID:-—}"
printf "  │ %-28s │ %-40s │\n" "ECR repository"  "${ECR_REPO:-—}"
[[ $DRY_RUN -eq 1 ]]  && printf "  │ %-28s │ %-40s │\n" "Dry-run"         "YES — no AWS calls"
[[ $SKIP_ECR -eq 1 ]] && printf "  │ %-28s │ %-40s │\n" "ECR build"       "SKIPPED"
printf "  └──────────────────────────────┴──────────────────────────────────────────┘\n"

printf "\n  ${BOLD}Proceed? [y/N]${NC}  "
read -r CONFIRM
if [[ "${CONFIRM,,}" != "y" ]]; then
  printf "\n  Aborted.\n\n"; exit 0
fi

# ── Step 4: Run pipeline ──────────────────────────────────────────────────────
hdr "Step 4/4  —  Running pipeline"
printf "\n"

if [[ "$MODE" == "local" ]]; then
  # ── Local mode: run all stages via run_local_pipeline.py ──────────────────
  SOURCE_TYPE_VAL="${SOURCE_TYPE:-s3_csv}"
  LABEL_VAL="${LABEL_COL:-is_fraud}"
  QUERY_VAL="${QUERY_FILE:-queries/model-data.sql}"

  EXPECTED_COLS=()
  while IFS= read -r col; do
    col=$(printf '%s' "$col" | sed 's/^[[:space:]]*-[[:space:]]*//' | xargs)
    [[ -n "$col" ]] && EXPECTED_COLS+=("--expected-column" "$col")
  done < <(awk '/expected_columns:/,/^[^ ]/' "$CONFIG_FILE" \
             | grep '^\s*-' | head -20)

  DRIFT_COLS=()
  while IFS= read -r col; do
    col=$(printf '%s' "$col" | sed 's/^[[:space:]]*-[[:space:]]*//' | xargs)
    [[ -n "$col" ]] && DRIFT_COLS+=("--drift-column" "$col")
  done < <(awk '/drift_columns:/,/^[^ ]/' "$CONFIG_FILE" \
             | grep '^\s*-' | head -20)

  CMD=(
    python3 "${SCRIPT_DIR}/orchestration/run_local_pipeline.py"
    --query-file   "$QUERY_VAL"
    --source-type  "$SOURCE_TYPE_VAL"
    --label-column "$LABEL_VAL"
    --raw-uri      "$(yaml_get "$CONFIG_FILE" "raw_uri")"
    --etl-uri      "$(yaml_get "$CONFIG_FILE" "etl_uri")"
    --feature-uri  "$(yaml_get "$CONFIG_FILE" "feature_uri")"
    --processed-uri "$(yaml_get "$CONFIG_FILE" "processed_uri")"
    --model-dir    "$(yaml_get "$CONFIG_FILE" "model_dir")"
    --metrics-dir  "$(yaml_get "$CONFIG_FILE" "metrics_dir")"
    --feature-log-path "$(yaml_get "$CONFIG_FILE" "feature_log_path")"
    --data-pull-engine pandas
    --min-minority-ratio 0.01
    "${EXPECTED_COLS[@]+"${EXPECTED_COLS[@]}"}"
    "${DRIFT_COLS[@]+"${DRIFT_COLS[@]}"}"
  )

  [[ $SKIP_DATA_PULL -eq 1 ]] && CMD+=(--skip-data-pull)

  CHAMPION_AUC=$(yaml_get "$CONFIG_FILE" "champion_auc")
  [[ -n "$CHAMPION_AUC" ]] && CMD+=(--champion-auc "$CHAMPION_AUC")

  info "Command: ${CMD[*]}"
  printf "\n"
  "${CMD[@]}"

  printf "\n"
  ok "Local pipeline complete."
  info "Artifacts written to outputs/model/  outputs/metrics/  outputs/model_validation.json"
  printf "\n"
  info "Next step — package and register:"
  printf "  python stages/deployment/package_model.py \\\n"
  printf "    --model-dir outputs/model \\\n"
  printf "    --output-path outputs/model.tar.gz \\\n"
  printf "    --s3-uri \"%s\"\n\n" "$(yaml_get "$CONFIG_FILE" "model_artifact_s3_uri")"
  printf "  python stages/deployment/register_model.py \\\n"
  printf "    --model-package-group \"%s\" \\\n" "${MODEL_GROUP:-fraud-transaction-model}"
  printf "    --model-artifact-s3-uri \"%s\" \\\n" "$(yaml_get "$CONFIG_FILE" "model_artifact_s3_uri")"
  printf "    --inference-image-uri \"<ecr-image-uri>\"\n\n"

else
  # ── Airflow mode: bootstrap (ECR build + register + trigger) ──────────────
  info "Command: python3 scripts/bootstrap_ml_pipeline.py --config $CONFIG_FILE ${PASSTHROUGH_ARGS[*]+"${PASSTHROUGH_ARGS[*]}"}"
  printf "\n"

  python3 "${SCRIPT_DIR}/scripts/bootstrap_ml_pipeline.py" \
    --config "$CONFIG_FILE" \
    "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

  printf "\n"
  ok "Pipeline submitted to Airflow DAG: ${DAG_ID:-improved_automl_pipeline}"
  info "Monitor at: Airflow UI → DAGs → ${DAG_ID:-improved_automl_pipeline}"
  info "Model group: ${MODEL_GROUP:-fraud-transaction-model} (pending manual approval after run)"
fi
