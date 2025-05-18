
import matplotlib.pyplot as plt
import torch

def visualize_result(img_path, detections,ind):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(img_path))
    plt.axis("off")
    for detection in detections:
        x1, y1, x2, y2,m = detection
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00FF00", linewidth=3
            )
        )
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00FF00", linewidth=1)
        )
        # ax.text(
        #     x1 + 5,
        #     y1 - 18,
        #     "{:.2f}".format(cor),
        #     bbox=dict(facecolor="#4CAF50", linewidth=0),
        #     fontsize=20,
        #     color="white",
        # )
    plt.tight_layout()
    # fig.savefig(img_path.replace("gallery", "result"))
    ind = str(ind)
    # i = str(i)
    lu = "E:/hahh/" + ind  +"-"+ "ours.jpg"
    fig.savefig(lu)
    # plt.show()
    plt.close(fig)


def visualize_result1(img_path, detections,ind):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(img_path))
    plt.axis("off")
    for detection in detections:
        x1, y1, x2, y2,m = detection
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#FF0000", linewidth=3
            )
        )
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#FF0000", linewidth=1)
        )
        # ax.text(
        #     x1 + 5,
        #     y1 - 18,
        #     "{:.2f}".format(score),
        #     bbox=dict(facecolor="#4CAF50", linewidth=0),
        #     fontsize=20,
        #     color="white",
        # )
    plt.tight_layout()
    # fig.savefig(img_path.replace("gallery", "result"))
    ind = str(ind)
    lu = "E:/compare/" + ind  + "seq.jpg"
    fig.savefig(lu)
    # plt.show()
    plt.close(fig)
def visualize_query(img_path, detections,score,id):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(img_path))
    plt.axis("off")
    for detection in detections:
        x1, y1, x2, y2 = detection
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#FFFF00", linewidth=3
            )
        )
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=1)
        )
        # ax.text(
        #     x1 + 5,
        #     y1 - 18,
        #     "{:.2f}".format(score),
        #     bbox=dict(facecolor="#4CAF50", linewidth=0),
        #     fontsize=20,
        #     color="white",
        # )
    plt.tight_layout()
    score = str(score)
    lu = "E:/hahh/"+score+"query.jpg"
    fig.savefig(lu)
    # plt.show()
    plt.close(fig)



import json

# filename = "E:/任务/论文/计算机视觉/计算机视觉大作业/计算机视觉project 张雯欣 Z21070225/code/resultsce.json"
filename = "E:/person_search/NEW_SOLIDER/vis/results.json"
filename1 = "C:/Users/shijie/Desktop/论文1写_投/vis/COAT_results.json"

with open(filename) as f:
    # 以字典的形式存储到all_eq_data中
    all = json.load(f)
with open(filename1) as f1:
    # 以字典的形式存储到all_eq_data中
    all1 = json.load(f1)

# readable_file="csv\mapping_global_data_sets\data/readable_eq_data.json"
# with open(readable_file,'w') as f:
#     #写入数据，参数 indent=4让 dump()使用与数据结构匹配的缩进量来设置数据的格式
#     json.dump(all_eq_data,f,indent=4)
imgpath = all['image_root']
all = all['results']
imgpath1 = all1['image_root']
all1 = all1['results']

# 提取信息
img,roi = [], []
img1,roi1 = [], []


ind = 0
for eq_dict,eq_dict1 in zip(all,all1):
    # print(eq_dict)
    query = eq_dict['query_img']
    queryroi =eq_dict['query_roi']
    queryroi = torch.tensor(queryroi, device='cpu')
    queryroi = queryroi.unsqueeze(dim=0)
    ind +=1
    id = str(query)
    query_gt = eq_dict['query_gt']
    gallery = eq_dict['gallery']
    lu = 'E:/任务/论文/计算机视觉/code/jichu/PRW-v16.04.20/frames/' + query
    gallery1 = eq_dict1['gallery']

    i = 0
    for index,index1 in zip(gallery,gallery1):
        i=i+1
        img =index['img']
        getroi = index['roi']
        getroi = torch.tensor(getroi, device='cpu')
        getroi = getroi.unsqueeze(dim=0)

        # roi = roi.append(getroi)
        score = index['score']
        correct = index['correct']

        img1 = index1['img']
        getroi1 = index1['roi']
        getroi1 = torch.tensor(getroi1, device='cpu')
        getroi1 = getroi1.unsqueeze(dim=0)

        # roi = roi.append(getroi)
        score1 = index1['score']
        correct1 = index1['correct']

        data = 'E:/任务/论文/计算机视觉/code/jichu/PRW-v16.04.20/frames'+'/'+img
        data1 = 'E:/任务/论文/计算机视觉/code/jichu/PRW-v16.04.20/frames'+'/'+img1
        if i ==1 and correct1==0 and correct==1:
        # if ind ==15 or ind ==23 or ind ==209 or ind ==224 or ind ==355or ind ==448 or ind ==457 or ind ==463or ind ==504  :
            visualize_query(lu, queryroi, ind,id)
            visualize_result(data,getroi,ind)
            visualize_result1(data1,getroi1,ind)




# filename = "E:/任务/论文/计算机视觉/计算机视觉大作业/计算机视觉project 张雯欣 Z21070225/code/results.json"
# # filename = "E:/任务/论文/计算机视觉/检索/code/prwresults.json"
#
# with open(filename) as f:
#     # 以字典的形式存储到all_eq_data中
#     all = json.load(f)
#
# # readable_file="csv\mapping_global_data_sets\data/readable_eq_data.json"
# # with open(readable_file,'w') as f:
# #     #写入数据，参数 indent=4让 dump()使用与数据结构匹配的缩进量来设置数据的格式
# #     json.dump(all_eq_data,f,indent=4)
# imgpath = all['image_root']
# all = all['results']
#
#
# # 提取信息
# img,roi = [], []
# img1,roi1 = [], []
#
#
# ind = 0
# for eq_dict in all:
#     # print(eq_dict)
#     query = eq_dict['query_img']
#     queryroi =eq_dict['query_roi']
#     queryroi = torch.tensor(queryroi, device='cpu')
#     queryroi = queryroi.unsqueeze(dim=0)
#     ind +=1
#     id = str(query)
#     query_gt = eq_dict['query_gt']
#     gallery = eq_dict['gallery']
#     zuo = queryroi[:,:2]
#     you = queryroi[:,2:]
#     querywh = you-zuo
#     score = querywh[0][1]*querywh[0][0]
#     print(score)
#     print(score / 100)
#     lu = 'E:/任务/论文/计算机视觉/计算机视觉大作业/计算机视觉project 张雯欣 Z21070225/code/data/CUHK-SYSU/Image/SSM/' + query
#     # lu = 'E:/任务/论文/计算机视觉/code/jichu/PRW-v16.04.20/frames/' + query
#
#
#     i = 0
#     for index in gallery:
#         i=i+1
#         img =index['img']
#         getroi = index['roi']
#         getroi = torch.tensor(getroi, device='cpu')
#         getroi = getroi.unsqueeze(dim=0)
#
#
#
#         zuo = getroi[:, :2]
#         you = getroi[:, 2:4]
#         getwh = you - zuo
#         score = getwh[0][1] * getwh[0][0]
#         print(score)
#         print(score/100)
#
#
#         # roi = roi.append(getroi)
#         score = index['score']
#         correct = index['correct']
#         # if ind ==1:
#         #     print(score)
#
#
#         # data = 'E:/任务/论文/计算机视觉/code/jichu/PRW-v16.04.20/frames/'+img
#         data = 'E:/任务/论文/计算机视觉/计算机视觉大作业/计算机视觉project 张雯欣 Z21070225/code/data/CUHK-SYSU/Image/SSM'+'/'+img
#         # if ind >9:
#         #     if i ==1:# and  correct==0:
#         # # if ind ==15 or ind ==23 or ind ==209 or ind ==224 or ind ==355or ind ==448 or ind ==457 or ind ==463or ind ==504  :
#         #         visualize_query(lu, queryroi, ind,id)
#         #     visualize_result(data,getroi,ind,i,correct)
#


