import yaml

# 类别文件 classes.txt（每行一个类别名）
classes_file = './data/classes.txt'

# 读取类别名
with open(classes_file, 'r', encoding='utf-8') as f:
    names = [line.strip() for line in f.readlines() if line.strip()]

# 构建 data.yaml 配置字典
data_yaml = {
    'train': './data/train.txt',
    'val': './data/val.txt',
    'nc': len(names),
    'names': names
}

# 保存为 data.yaml
with open('dataset/data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print("成功生成 data.yaml 文件，内容如下：")
print(yaml.dump(data_yaml, allow_unicode=True))
