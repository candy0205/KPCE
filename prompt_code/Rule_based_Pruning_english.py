from collections import Counter
from string import punctuation

dicts={i:'' for i in punctuation}
punc_table=str.maketrans(dicts)

        
def compare(s, t):
    s_l = [e.lower() for e in s]
    t_l = [e.lower() for e in t]
    return Counter(s_l) == Counter(t_l)

def rule_check(info, con_ls):
    con_ls = [con for con in con_ls if len(con) > 2]
    r = con_ls[:]
    l = len(con_ls)
    
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        if  'also' in con_word or 'called' in con_word or 'are' in con_word or 'is' in con_word:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
    
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        
        if len(con_ls[i]) <= 2:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
        
        if len(con_word) > 30:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
    
    r = list(set(r))
     
    if compare(r, con_ls):
        return r,0
    else:
        return r,1

if __name__ == '__main__':
    dataset = {}
    with open('original_result_probase.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, raw_con, con = data.replace('\n','').split('\t')
            con = con.split(',')
            dataset[info] = [raw_con,con]
    f.close()
    
    with open('original_result_probase_easysee.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            f1.write(info)
            f1.write('-----')
            f1.write(dataset[info][0])
            f1.write('-----')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()

    k = 0
    for info in dataset.keys():
        con_ls = dataset[info][1]
        r,flag = rule_check(info, con_ls)
        if flag != 0:
            k = k + 1
            print(dataset[info][0], con_ls, r)
            dataset[info][1] = r
    
    with open('rule_result_probase.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():
            f1.write(info)
            f1.write('\t')
            f1.write(dataset[info][0])
            f1.write('\t')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()
    
    with open('rule_result_probase_easysee.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            f1.write(info)
            f1.write('-----')
            f1.write(dataset[info][0])
            f1.write('-----')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()