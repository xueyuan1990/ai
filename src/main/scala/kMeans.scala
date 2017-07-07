/**
  * Created by xueyuan on 2017/5/4.
  */
package com.test


import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.linalg.Vector

object kMeans {
  var sc: SparkContext = null
  var candidate_df: DataFrame = null
  val path2 = "/tmp/xueyuan2/1493880167163/model"

  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    println("***********************start*****************************")
    val sparkConf: SparkConf = new SparkConf().setAppName("platformTest")
    sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    println("***********************sc*****************************")
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    val hiveContext: HiveContext = new HiveContext(sc)
    println("***********************hive*****************************")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    //get condidate_rdd
    load_data(hiveContext, sc)
    diff()
  }

  def load_data(hiveContext: HiveContext, sc: SparkContext): Unit = {
    val sql_1 = "select * from project.kmeans_features"
    val df = hiveContext.sql(sql_1)
    val all_rdd = df.map(r => (r.getString(0), r.getString(1).split(",")))
    val all_row = all_rdd.map(r => Row(r._1, r._2(0).toDouble, r._2(1).toDouble))
    //通过编程方式动态构造元数据
    val structType = StructType(Array(
      StructField("label", StringType, true),
      StructField("f1", DoubleType, true),
      StructField("f2", DoubleType, true)
    ))
    val sqlContext = new SQLContext(sc)

    //进行RDD到DataFrame的转换
    candidate_df = sqlContext.createDataFrame(all_row, structType)
  }

  def training(): Array[Vector] = {
    val colArray = Array("f1", "f2")
    val assembler = new VectorAssembler().setInputCols(colArray).setOutputCol("features")
    val vec_df = assembler.transform(candidate_df)
    val kmeans = new KMeans().setK(8).setTol(0.0001).setInitSteps(2).setMaxIter(51).setInitMode("k-means||")
    kmeans.setFeaturesCol("features")
    val model = kmeans.fit(vec_df)
    val center = model.clusterCenters
    for (c <- center) println("***********************c=" + c + "*****************************")
    center
  }

  def load_model2(): Array[Vector] = {
    val model2 = KMeansModel.load(path2)
    val center2 = model2.clusterCenters
    for (c2 <- center2) println("***********************c2=" + c2 + "*****************************")
    center2
  }

  def diff(): Unit = {
    val center = training()
    val center2 = load_model2()
    val k = center.size
    for (i <- 0 until k) {
      val c = center(i).toArray
      val c2 = center2(i).toArray
      sim(c, c2)
    }

  }

  def sim(weight: Array[Double], weight2_resorted: Array[Double]): Unit = {
    //caculate
    //对公式部分分子进行计算
    println("***********************weight*****************************")
    for (w <- weight) print(w + " ")
    println("***********************weight2*****************************")
    for (w2 <- weight2_resorted) print(w2 + " ")
    println("")
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
    //    println("***********************temp1=" + temp1 + ";temp2=" + temp2 + "*****************************")
    //    println("***********************member=" + member + ";denominator=" + denominator + "*****************************")
    val sim = member / denominator
    println("***********************sim=" + sim + "*****************************")
  }


}
