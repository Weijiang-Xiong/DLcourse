modelPath = './myProject/'
with open( modelPath +'trainLoss.txt', 'w') as f:
    for item in [1,2,3,4,5]:
        f.write("%s\n" % item)