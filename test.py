import url
import re

# text = """
# The regex patterns in this gist are intended only to match web URLs -- http,
# https, and naked domains like "example.com". For a pattern that attempts to
# match all URLs, regardless of protocol, see: https://gist.github.com/gruber/249502
# """
# text2=":'("

reg="""(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)"""
reg1="""(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))""" #Smile
reg2="""(:\s?D|:-D|x-?D|X-?D)""" #Laugh
reg3="""(;\s?\(|:-\(|\)\s?:|\)-;)""" #Wink
reg4="""(<3|:\*)""" #Love
reg5="""(:\s?\(|:-\(|\)\s?:|\)-:)""" #Sad
reg6="""(:,\(|:\'\(|:"\()""" #Cry

reg7="""(\.+|\?+|!+)"""
t=";)"
				
print(re.findall(reg3,t))
# l=[[1,3],[4,5]]
# for i in l:
# 	for j in i:
# 		j=0
# print(l)