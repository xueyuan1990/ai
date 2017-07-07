package com.test

/**
  * Created by xueyuan on 2017/4/24.
  */

import java.util.regex.Pattern
import java.io._
import org.apache.spark.ml.classification.{LogisticRegression}
import org.apache.hadoop.fs.{FSDataOutputStream, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{Row, SQLContext}

import scala.collection.{Map, mutable}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object logisticRegression {
  var sc: SparkContext = null
  var candidate_rdd: RDD[(Double, SparseVector)] = null
  var feature_index_map: Map[String, Long] = null
  var weight : Array[Double] = null
  val logistic_regression_path = "/tmp/xueyuan/logistic_regression/model"
  val logistic_regression_path2 = "/tmp/xueyuan2/1493777256545/model"
  val logistic_regression_path3 = "/tmp/xueyuan2/1493782245548/s/model"
  val feature_index_rdd_file = "/tmp/xueyuan/logistic_regression/feature_id_index.txt"
  val feature_index_rdd_file2 = "/tmp/xueyuan2/1493777256545/feature_id_index.txt"
  val feature_index_rdd_file3 = "/tmp/xueyuan2/1493782245548/s/feature_id_index.txt"
  val weight_file = "/tmp/xueyuan/logistic_regression/weight.txt"
  val weight2_file = "/tmp/xueyuan2/1493777256545/weight2.txt"
  val weight3_file = "/tmp/xueyuan2/1493782245548/s/weight2.txt"
  val weight2_sorted_file = "/tmp/xueyuan2/1493777256545/weight2_sorted.txt"
  val weight3_sorted_file = "/tmp/xueyuan2/1493782245548/s/weight2_sorted.txt"
  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    val sparkConf: SparkConf = new SparkConf().setAppName("platformTest")
    sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    val hiveContext: HiveContext = new HiveContext(sc)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    load_data(hiveContext, sc)
    training()
//    loadModel()
    diff()
  }

//  def loadModel(): Unit = {
//    val model = LogisticRegressionModel.load(sc, logistic_regression_path)
//    println("***********************" + model.toString() + "*****************************")
//    println("***********************" + model.toPMML() + "*****************************")
//    val weight = model.weights.toArray
//    for(item<-weigth) print(item+" ")
    //    val webFile=Source.fromURL(feature_index_rdd_path)
//  }

  def training(): Unit = {
    println("***********************training begin*****************************")
    val data = candidate_rdd.map(r => new LabeledPoint(r._1, r._2))
    println("***********************LabeledPoint*****************************")
    // Split data into training (60%) and test (40%).
//    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
//    val training = splits(0).cache()
//    val test = splits(1)
    val training = data
    println("***********************splits finished*****************************")
    // Run training algorithm to build the model

    val lr = new LogisticRegressionWithLBFGS().setIntercept(true).setNumClasses(2)
    lr.optimizer.setRegParam(0.6).setConvergenceTol(0.000001).setNumIterations(101).setUpdater(new SquaredL2Updater)
    val model = lr.run(training).setThreshold(0.5)

    println("***********************generage model*****************************")
//    // Compute raw scores on the test set.
//    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
//      val prediction = model.predict(features)
//      (prediction, label)
//    }
//    println("***********************get predictionAndLabels*****************************")
//
//    // Get evaluation metrics.
//    val metrics = new MulticlassMetrics(predictionAndLabels)
//    println("***********************get metrics*****************************")
//    val precision = metrics.precision
//    println("Precision = " + precision)

    // Save and load model
    weight = model.weights.toArray
    val output = new Path(logistic_regression_path)
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    if (hdfs.exists(output)) hdfs.delete(output, true)
    model.save(sc, logistic_regression_path)


  }


  def load_data(hiveContext: HiveContext, sc: SparkContext): Unit = {
    val sql_1 = "select * from project.logistic_regression_features"
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    val data_rdd = df.map(r => (r.getDouble(1), r.getString(2).split(",")))
    val feature_index_rdd = data_rdd.flatMap(r => r._2).distinct().zipWithIndex()
    feature_index_map = feature_index_rdd.collectAsMap()
    val f_i_map = feature_index_map

//    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
//    val path = new Path(feature_index_rdd_file)
//    val outputstream = hdfs.create(path, true)
//
//
//    for (item <- feature_index_map) {
//      outputstream.writeBytes(item._1.toString + " " + item._2.toString + "\r\n")
//    }
//    outputstream.close()

    val feature_index_map_size = feature_index_map.size
    println("***********************feature_index_map size =" + feature_index_map.size + "*****************************")
    val lable_feature_rdd = data_rdd.map(r => {
      val feature = r._2
      val size = feature.size
      val value_array = Array.fill(size)(1.0)
      val index_array = new ArrayBuffer[Int]()
      for (item <- feature) {
        val i = f_i_map(item)
        val index = i.toInt
        index_array += index
      }


      val sp = new SparseVector(feature_index_map_size, index_array.toArray.sortWith(_ < _), value_array)
      (r._1, sp)

    })
    lable_feature_rdd.map(r => r._1)
    println("***********************generate sp finished*****************************")
    candidate_rdd = lable_feature_rdd
  }

  def diff(): Unit = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    println("***********************hdfs*****************************")
    //load feature_index_map
//    val path: Path = new Path(feature_index_rdd_file3)
//    val reader = new BufferedReader(new InputStreamReader(hdfs.open(path),"utf-8"))
//    println("***********************reader*****************************")
//    var feature_index_map3: Map[String, Long] = new mutable.HashMap[String,Long]()
//    var line = reader.readLine()
//    while (line != null) {
//      val feature_index = line.split(" ")
//      feature_index_map3 += (feature_index(0) -> feature_index(1).toLong)
//      line = reader.readLine()
//    }
//    println("***********************feature_index_map3.size="+feature_index_map3.size+"*****************************")
//load weight1
//    val model = org.apache.spark.ml.classification.LogisticRegressionModel.load(logistic_regression_path3)
//    val weight = model.weights.toArray
//
//    println("***********************weight.size="+weight.size+"*****************************")
    //load feature_index_map2
    val path2 = new Path(feature_index_rdd_file2)
    val reader2 = new BufferedReader(new InputStreamReader(hdfs.open(path2),"utf-8"))
    println("***********************reader2*****************************")
    var feature_index_map2: Map[String, Long] = new mutable.HashMap[String,Long]()
    var line2 = reader2.readLine()
    while (line2 != null) {
      val feature_index = line2.split(" ")
      feature_index_map2 += (feature_index(0) -> feature_index(1).toLong)
      line2 = reader2.readLine()
    }
    println("***********************feature_index_map2.size="+feature_index_map2.size+"*****************************")

    //load weight2
    val model2 = org.apache.spark.ml.classification.LogisticRegressionModel.load(logistic_regression_path2)
    val weight2 = model2.weights.toArray
    println("***********************weight2.size="+weight2.size+"*****************************")
    //resort weight2
    val weight2_resorted: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    println("***********************weight2_resorted start*****************************")
    val index_feature_map = feature_index_map.map(r=>(r._2,r._1))
    for(i<-0 until index_feature_map.size){
      val feature = index_feature_map(i)
      val index2 = feature_index_map2(feature).toInt
      weight2_resorted += weight2(index2)
    }

    //save weight
    val outputstream = hdfs.create(new Path(weight_file), true)
    for (item <- weight) {
      outputstream.writeBytes(item + "\r\n")
    }
    outputstream.close()
    //save weight2
    val outputstream2 = hdfs.create(new Path(weight2_file), true)
    for (item <- weight2) {
      outputstream2.writeBytes(item + "\r\n")
    }
    outputstream2.close()
    //save weight2_resorted
    val outputstream2_sorted = hdfs.create(new Path(weight2_sorted_file), true)
    for (item <- weight2_resorted) {
      outputstream2_sorted.writeBytes(item + "\r\n")
    }
    outputstream2_sorted.close()
    sim(weight,weight2_resorted.toArray)

  }

  def sim(weight:Array[Double],weight2_resorted:Array[Double]):Unit= {
    //caculate
    //对公式部分分子进行计算
    //    println("***********************weight*****************************")
    //    for(w<-weight) print(w+" ")
    //    println("***********************weight2*****************************")
    //    for(w2<-weight2_resorted) print(w2+" ")
    //    println("")
    val member = weight.zip(weight2_resorted).map(d => d._1 * d._2).reduce(_ + _)
    //求出分母第一个变量值
    val temp1 = math.sqrt(weight.map(num => {
      math.pow(num, 2)
    }).reduce(_ + _))
    //求出分母第二个变量值
    val temp2 = math.sqrt(weight2_resorted.map(num => {
      math.pow(num, 2)
    }).reduce(_ + _))
    //求出分母
    val denominator = temp1 * temp2
    //进行计算
    println("***********************temp1="+ temp1 +";temp2="+temp2+"*****************************")
    println("***********************member="+ member +";denominator="+denominator+"*****************************")
    val sim = member / denominator
    println("***********************sim="+ sim +"*****************************")
  }


}
