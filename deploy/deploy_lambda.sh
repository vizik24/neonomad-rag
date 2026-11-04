#!/usr/bin/env bash
# Build the image in the format Lambda accepts, push it to ECR and point the function at it.
#
#   deploy/deploy_lambda.sh
#
# Settings (environment variables, API key, database URL) stay as they are on the function; only the code
# changes. Re-run ingest.py against Supabase separately when docs/md changes (see README, "Deployment").
set -euo pipefail
cd "$(dirname "$0")/.."

REGION=${REGION:-eu-north-1}
FUNCTION=${FUNCTION:-neonomad-support}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
TAG="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo -dirty)"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

# Lambda only accepts a single-platform Docker v2 image: Docker Desktop's default (an OCI index with
# provenance and SBOM attestations) is rejected with "image manifest ... media type is not supported".
docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
  --output "type=image,name=${REGISTRY}/neonomad-support:${TAG},push=true,oci-mediatypes=false" .

aws lambda update-function-code --region "$REGION" --function-name "$FUNCTION" \
  --image-uri "${REGISTRY}/neonomad-support:${TAG}" --query LastUpdateStatus --output text
aws lambda wait function-updated-v2 --region "$REGION" --function-name "$FUNCTION"

URL=$(aws lambda get-function-url-config --region "$REGION" --function-name "$FUNCTION" --query FunctionUrl --output text)
echo "Deployed ${TAG}. Health: $(curl -s -m 90 "${URL}health")"
