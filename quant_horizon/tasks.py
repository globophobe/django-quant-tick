"""Invoke tasks for deployment."""

import sys
from pathlib import Path

from invoke import task


@task
def deploy(ctx):
    """Deploy quant_horizon to Cloud Run.

    Queries Django DB for is_production=True MLArtifact,
    extracts GCS path, and deploys Cloud Run service with MODEL_PATH env var.
    """
    # Add Django project to path for DB access
    django_path = Path(__file__).parent.parent / "django-quant-tick"
    sys.path.insert(0, str(django_path))

    # Setup Django
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo.settings.production")

    try:
        import django

        django.setup()
    except Exception as e:
        print(f"WARNING: Could not setup Django: {e}")
        print("Proceeding with manual MODEL_PATH...")
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            print("ERROR: No production model found and MODEL_PATH not set.")
            print("Set MODEL_PATH environment variable or ensure Django is accessible.")
            sys.exit(1)
    else:
        # Query for production model
        from quant_tick.models import MLArtifact

        artifact = MLArtifact.objects.filter(is_production=True).first()
        if not artifact:
            print("ERROR: No production model found (is_production=True)")
            print("Run: python manage.py train_model <id> --set-production")
            sys.exit(1)

        model_path = artifact.gcs_path
        if not model_path:
            print(f"ERROR: Production artifact {artifact.id} has no GCS path")
            sys.exit(1)

        print(f"Found production model: {artifact.id}")
        print(f"GCS path: {model_path}")

    # Get GCP project
    result = ctx.run("gcloud config get-value project", hide=True)
    project_id = result.stdout.strip()

    if not project_id:
        print("ERROR: No GCP project configured")
        print("Run: gcloud config set project <project-id>")
        sys.exit(1)

    print(f"Deploying to project: {project_id}")

    # Build and push container
    image = f"gcr.io/{project_id}/quant-horizon:latest"
    print(f"Building image: {image}")

    # Build from parent directory (to include quant_core)
    ctx.run(f"docker build -f quant_horizon/Dockerfile -t {image} .", echo=True)
    ctx.run(f"docker push {image}", echo=True)

    # Deploy to Cloud Run
    print(f"Deploying to Cloud Run with MODEL_PATH={model_path}")

    deploy_cmd = (
        f"gcloud run deploy quant-horizon "
        f"--image={image} "
        f"--platform=managed "
        f"--region=asia-northeast1 "
        f"--set-env-vars=MODEL_PATH={model_path} "
        f"--allow-unauthenticated "
        f"--memory=2Gi "
        f"--cpu=2 "
        f"--min-instances=0 "
        f"--max-instances=10 "
        f"--timeout=60s "
        f"--startup-probe-initial-delay-seconds=10 "
        f"--startup-probe-timeout-seconds=5 "
        f"--startup-probe-period-seconds=5 "
        f"--startup-probe-failure-threshold=3 "
        f"--startup-probe-http-get-path=/health "
        f"--liveness-probe-initial-delay-seconds=10 "
        f"--liveness-probe-timeout-seconds=5 "
        f"--liveness-probe-period-seconds=10 "
        f"--liveness-probe-failure-threshold=3 "
        f"--liveness-probe-http-get-path=/health"
    )

    ctx.run(deploy_cmd, echo=True)

    print("\n✅ Deployment complete!")
    print("\nGet service URL:")
    print("  gcloud run services describe quant-horizon --region=asia-northeast1 --format='value(status.url)'")


@task
def test_local(ctx):
    """Test service locally with a mock model."""
    print("Starting local test server...")
    print("Note: This requires a valid MODEL_PATH or will fail startup")
    print("\nSet MODEL_PATH to test:")
    print("  export MODEL_PATH=gs://your-bucket/path/to/model.joblib")
    print("\nThen run:")
    print("  uvicorn quant_horizon.main:app --reload --port 8000")
