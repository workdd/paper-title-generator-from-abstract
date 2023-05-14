FROM public.ecr.aws/lambda/python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip
RUN yum install git -y
RUN git clone https://github.com/workdd/recommand-title.git

RUN pip install -r recommand-title/requirements.txt
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
RUN aws s3 cp $S3_MODELS ${LAMBDA_TASK_ROOT} --recursive --profile $account
RUN cp recommand-title/app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]