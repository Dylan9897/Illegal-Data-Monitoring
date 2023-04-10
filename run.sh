docker run -d -p 5011:5006 --name ood2 --hostname ood2 -v /model/handx/multi_label_task/classifier_multi_label_textcnn/deploy/log:/log --privileged=true ood:v1.3

