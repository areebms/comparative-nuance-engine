# login first if needed
aws ecr get-login-password --region us-west-2 \
 | docker login --username AWS --password-stdin $AWS_URI_PREFIX

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --no-cache \
  --load \
  -t $AWS_URI_PREFIX/$AWS_ECR_REPO/lambda-generate-kvector:0.1 \
  -f functions/generate-kvector/Dockerfile .

docker run --rm -it --env-file .env --entrypoint python \
  $AWS_URI_PREFIX/$AWS_ECR_REPO/lambda-generate-kvector:0.1 \
  generate_kvector.py --platform-name gutenberg --platform-id 60411

docker push $AWS_URI_PREFIX/$AWS_ECR_REPO/lambda-generate-kvector:0.1

DIGEST=$(aws --no-cli-pager ecr describe-images \
  --repository-name $AWS_ECR_REPO/lambda-generate-kvector \
  --image-ids imageTag=0.1 \
  --query 'imageDetails[0].imageDigest' --output text)

aws --no-cli-pager lambda update-function-code \
  --function-name generate-kvector \
  --image-uri $AWS_URI_PREFIX/$AWS_ECR_REPO/lambda-generate-kvector@$DIGEST