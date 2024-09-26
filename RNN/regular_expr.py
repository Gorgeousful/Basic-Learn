# 学习正则表达式
import re
#match 从字符串起始处开始匹配 最多返回一个结果
string = 'MY_PHONE my_phone'  # 要匹配的字符串
pattern = r'my_\w+'  # 模式字符串
match = re.match(pattern, string, re.I)  # 匹配字符串，设置为不区分大小写
print('匹配值的起始位置:', match.start())
print('匹配值的结束位置:', match.end())
print('匹配位置的元组:', match.span())
print('要匹配的字符串:', match.string)
print('匹配数据:', match.group())
#search 从整个字符串匹配 直返回遇到的第一个  最多返回一个结果
string = 'abcMY_PHONE my_phone'
pattern = r'my_\w+'
match = re.search(pattern, string, re.I)
print(match.group())
#findall 从整个字符串中找到所有匹配的
string = 'abcMY_PHONE my_phone'
pattern = r'my_\w+'
match = re.findall(pattern, string, re.I)
print(match)
#sub 字符串替换
string = '姓名 - 张三,学历 - 博士,性别 - 男'
result = re.sub(r' ','-', string)
print(result)
#split 分割字符
string = '姓名 - 张三,学历 - 博士,性别 - 男'
result = re.split(r' - |,', string)
print(result)