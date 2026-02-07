FROM public.ecr.aws/lambda/python:3.13

COPY functions/scrape/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY ./shared shared
COPY functions/scrape/src/ .
CMD ["app.handler"]
