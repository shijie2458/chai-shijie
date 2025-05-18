import json

# 读取原始JSON文件
with open('CGPS-main/show/input.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 重新排序id，从11100开始
new_id = 11000
for item in data:
    item['id'] = new_id
    new_id += 1

# 将修改后的数据写入新文件，输出为一行且有空格
with open('CGPS-main/show/output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, separators=(', ', ': '))

print("ID已重新排序，并保存到output.json")

