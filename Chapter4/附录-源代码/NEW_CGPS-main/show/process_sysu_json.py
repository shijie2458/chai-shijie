import json

# 读取原始JSON文件
with open('CGPS-main/show/input.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 重新排序id，从11100开始
new_id = 53689
for item in data:
    item['id'] = new_id
    new_id += 1

# 将每个"image_id"的值加上4300
for item in data:
    if 'image_id' in item:  # 检查是否存在 'image_id' 键
        item['image_id'] += 4300  # 将 'image_id' 加上4300

# 将修改后的数据写入新文件，输出为一行且有空格
with open('CGPS-main/show/output_sysu.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, separators=(', ', ': '))

print("ID已重新排序，并保存到output.json")

