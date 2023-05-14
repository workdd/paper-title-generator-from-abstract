FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN aws s3 cp $S3_MODELS ${LAMBDA_TASK_ROOT} --recursive

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]