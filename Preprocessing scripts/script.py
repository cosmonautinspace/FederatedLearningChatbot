file = open('Sahil_Notebook_ADS.txt', 'r')

lines = file.read().split('.')
encodedLines = []

for line in lines:
    encodedLines.append(line.encode("utf-8"))

print(encodedLines)