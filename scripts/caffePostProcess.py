if __name__ == "__main__":
    count = 0
    with open("top1Accuracy.txt", "r") as fh:
        data = fh.readlines()
        data = [x.strip() for x in data]         
        for i in data:
            if(int(i) == 1):
                count += 1
                
    print(count / 500.0)