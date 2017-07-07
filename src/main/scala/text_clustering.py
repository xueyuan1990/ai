# -*- coding: UTF-8 -*-
__author__ = 'tanlingcheng'

import pydoop.hdfs as hdfs
import numpy as np

def find_cluster(clusters_in, article, threshold_in):
    id  = article[0]
    words = article[1]
    vec = article[2]
    max_sim = threshold_in
    max_index = -1
    for c in clusters_in:
        cluster_words = c[1].keys()
        article_words = article[1].keys()
        join_words_num = len(set(cluster_words) & set(article_words))
        c.append(join_words_num)
    clusters_in.sort(key=lambda x: x[3], reverse=True)
    for i,c in enumerate(clusters_in):
        index = i
        cluster_vec = c[2]
        cluster_vec_norm = np.linalg.norm(cluster_vec)
        sim = np.vdot(vec, cluster_vec)/cluster_vec_norm
        if(sim>threshold_in):
            max_index = index
            break
    if(max_index == -1):
        ids = id
        sum_words = words
        sum_vec = vec
        clusters_in.append([ids, sum_words, sum_vec])
    else:
        clusters_in[max_index][0].extend(id)
        temp = reduce(sum_tfidf, words.iteritems(), clusters_in[max_index][1])
        if len(temp)>50:
            temp2 = dict(sorted(temp.iteritems(), key=lambda x: x[1], reverse=True)[:50])
            clusters_in[max_index][1] = temp2
        else:
            clusters_in[max_index][1] = temp
        clusters_in[max_index][2] = clusters_in[max_index][2] + vec
    clusters_in = [c[:2] for c in clusters_in]

# x：topic exeisted，y: topic being merged
def sum_tfidf(x,y):
    w = y[0]
    if x.has_key(w):
        x[w] = x[w] + y[1]
    else:
        x[w] = y[1]
    return x

if __name__ == "__main__":
    td = "20170302"
    hdfs_file="/user/hive/warehouse/algo.db/lingcheng_label_docvec_with_blas/stat_date="
    hadoop_host = "hd-snn-1.meizu.gz"
    hadoop_port = 9000
    threshold = 0.9
    fs = hdfs.fs.hdfs(host=hadoop_host,port=int(hadoop_port))
    file_list_string = hdfs_file + td
    file_list_origin = fs.list_directory(file_list_string)
    file_list = [flo["path"] for flo in file_list_origin]
    id_vec = []
    for f in file_list:
        content = fs.open_file(f)
        for line in content:
            ori = line.split('\001')
            id = [ori[0]]
            words = dict(map(lambda x: (x.split(":")[0], float(x.split(":")[1])), ori[1].split(",")))
            vec = np.array(map(lambda x: float(x), ori[2].split(",")))
            id_vec.append((id, words, vec))
    clusters = []
    if np.vdot(id_vec[0][2], id_vec[1][2])>threshold:
        ids = id_vec[0][0].extend(id_vec[1][0]) # article ids contained in the topic
        words = reduce(sum_tfidf, id_vec[1][1].iteritems(), id_vec[0][1]) # keywords of topic
        sum_vec = id_vec[0][2] + id_vec[1][2]  # vector of topic
        clusters.append([ids, words, sum_vec])
    else:
        ids0 = id_vec[0][0]
        sum_words0 = id_vec[0][1]
        sum_vec0 = id_vec[0][2]
        ids1 = id_vec[1][0]
        sum_word1 = id_vec[1][1]
        sum_vec1 = id_vec[1][2]
        clusters.append([ids0, sum_words0, sum_vec0])
        clusters.append([ids1, sum_word1, sum_vec1])
    id_vec2 = id_vec[2:]
    global count
    count = 0
    for item in id_vec2:
        find_cluster(clusters, item, threshold)
        count += 1
        print "=============================== count is " + str(count) + "================================"
        print "******************************* cluster length is " + str(len(clusters)) + "***************************"
        for index ,c in enumerate(clusters):
            print index
            print c[0]
            for w in c[1].items():
                print w[0], w[1]
        if count>10000:
            break
    # fs = open("cluster.txt", "w")
    # for item in clusters:
    #     ids = ','.join(item[0])
    #     words = str(item[1])
    #     vecstr_list = [str(i) for i in list(item[2])]
    #     vec = ','.join(vecstr_list)
    #     genstr = ids + " " + words + " " + vec + "\n"
    #     fs.writelines(genstr)





