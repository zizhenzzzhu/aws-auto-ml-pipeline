# Deployment Options

Use ECR for both training and inference images. ECS is the simpler default for
managed real-time inference when Kubernetes is not otherwise required. EKS is a
better fit when the platform already standardizes on Kubernetes, needs custom
operators, or shares GPU/node-pool scheduling across workloads.

## ECS

Use `deployment/ecs/task_definition.template.json` as a starting point for a
Fargate task or service. Replace account IDs, regions, roles, image URI, and
model mount strategy before deployment.

## EKS

Use `deployment/eks/deployment.yaml` as a starting point for a Kubernetes
Deployment and Service. Replace image URI, service account, resource requests,
secrets, and model volume configuration before deployment.
