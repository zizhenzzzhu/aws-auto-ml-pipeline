param(
    [Parameter(Mandatory = $true)][string]$RepositoryName,
    [Parameter(Mandatory = $true)][string]$AccountId,
    [string]$Region = "us-west-2",
    [string]$Dockerfile = "containers/training/Dockerfile.train",
    [string]$Tag = "latest"
)

$ImageUri = "$AccountId.dkr.ecr.$Region.amazonaws.com/$RepositoryName`:$Tag"

aws ecr describe-repositories --repository-names $RepositoryName --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --repository-name $RepositoryName --region $Region
}

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
docker build -f $Dockerfile -t $ImageUri .
docker push $ImageUri
Write-Output $ImageUri
