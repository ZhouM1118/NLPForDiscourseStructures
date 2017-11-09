content_1 = []
content_2 = []
content_3 = []
result = {
  1: lambda x: content_1.append(x),
  2: lambda x: content_2.append(x),
  3: lambda x: content_3.append(x)
}
result[1]('1')
result[1]('11')
result[1]('111')
result[2]('2')
result[2]('22')
result[2]('222')
result[3]('3')
result[3]('33')
result[3]('333')

print(content_1)
print(content_2)
print(content_3)