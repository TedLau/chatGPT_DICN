import math


# todo: generate some random adjs and some graphs
# todo: first one thousand belong to the first day
# todo: make the node number as the graph labels
# todo: but how to generate such a graph?
cnt = []
def gen_adjs(num,begin,day):
    with open('/Users/tedlau/Desktop/IMDB-BINARY/raw/IMDB-BINARY/IMDB-BINARY_graph_indicator.txt', 'a+') as w:
        for i in range(begin,begin+num):
            w.writelines(str(day)+"\n")

    with open('/Users/tedlau/Desktop/IMDB-BINARY/raw/IMDB-BINARY/IMDB-BINARY_A.txt', 'a+') as w:
        global cnt

        for i in range(math.ceil((num)/2)+1):
            if i >= (num)/2 - 3:
                cnt.append(i + 1 + begin)
                cnt.append(i + 3 + begin)
                print("{}, {}".format(i+1+begin,i+3+begin),file=w)
                print("{}, {}".format(i+2+1+begin,i+1+begin),file=w)
            if i+1+begin != num-i-1+begin:
                cnt.append(i + 1 + begin, )
                cnt.append(num-i-1 + begin)
                print("{}, {}".format(i+1+begin,num-i-1+begin),file=w)
                print("{}, {}".format(num-i-1+begin,i+1+begin),file=w)
        print('final des is :',len(cnt))



if __name__ == '__main__':
    gen_adjs(100,0,1)
    print(cnt)
    cnt.sort()
    print(cnt)
    cnt = list(set(cnt))
    print(len(cnt))
    gen_adjs(100,100,2)
    gen_adjs(100,200,3)
    gen_adjs(200,300,4)
    gen_adjs(300,500,5)
    gen_adjs(100,800,6)
    gen_adjs(100,900,7)
    gen_adjs(100,1000,8)
