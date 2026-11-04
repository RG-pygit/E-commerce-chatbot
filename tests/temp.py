inputString = "hello hello i am here here"
 
outputDict = {}

s=0
h=inputString.split()
print((h))
print(type(h))
print(h[1])

for i in h:
    outputDict[i]=outputDict.get(i,0)+1
    # if i in outputDict:
    #     outputDict[i]+=1
    #     print(outputDict)
    # else:
    #     outputDict[i]=1
    #     print(outputDict)\
print(outputDict)

    




    