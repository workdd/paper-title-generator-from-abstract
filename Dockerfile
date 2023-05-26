FROM public.ecr.aws/lambda/python:3.11-preview

RUN yum install git -y
RUN git clone https://github.com/workdd/recommand-title.git

RUN pip install -r recommand-title/requirements.txt
RUN cp recommand-title/app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]