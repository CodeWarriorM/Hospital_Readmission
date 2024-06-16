#local API
run_api_local:
	uvicorn packages.fast_api:app --reload

#Docker local
build_container_local:
	docker build . -t $$IMAGE

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 $$IMAGE:dev

#Docker Deployment
allow_docker_push:
	gcloud auth configure-docker $$GCP_REGION-docker.pkg.dev

create_artifacts_repo:
	gcloud artifacts repositories create $$ARTIFACTSREPO --repository-format=docker \
		--location=$$GCP_REGION --description="Repository for storing images"


build_for_production:
	docker build -t  $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod .

push_image_production:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod

deploy_to_cloud_run:
	gcloud run deploy --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod --memory $$MEMORY --region $$GCP_REGION

#Streamlit
streamlit:
	-@streamlit run packages/app.py
