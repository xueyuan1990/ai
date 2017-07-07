package com.test

/**
  * Created by xueyuan on 2017/4/24.
  */


import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.ml.regression._

object linearRegression {
  var sc: SparkContext = null
  var candidate_df: DataFrame = null
  val path2 = "/tmp/xueyuan2/1493806519527"

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
    println("***********************load*****************************")
    sim(training(),load_model2())

  }

  //  def loadModel(): Unit = {
  //    val model = LinearRegressionModel.load(sc, linear_regression_path)
  //    println("***********************" + model.toString() + "*****************************")
  //    println("***********************" + model.toPMML() + "*****************************")
  //  }

  def training(): Array[Double] = {


    val colArray = Array("f1", "f2", "f3", "f4")
    val assembler = new VectorAssembler().setInputCols(colArray).setOutputCol("features")
    val vec_df = assembler.transform(candidate_df)

    val lr = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFitIntercept(true)
      .setStandardization(true)
      .setTol(0.000001)

    lr.setFeaturesCol("features").setLabelCol("label")
    // Fit the model
    val lrModel = lr.fit(vec_df)
    val weight = lrModel.weights.toArray
    for (w <- weight) println("*****************************w1=" + w + "*****************************")
    weight

  }


  def load_data(hiveContext: HiveContext, sc: SparkContext): Unit = {
    val sql_1 = "select * from project.linear_regression_features2"
    val df = hiveContext.sql(sql_1)
    //split all features
    val all_rdd = df.map(r => (r.getDouble(1), r.getString(2).split(",")))
    //get 100 candidates , only use 4 features
    val candidate_rdd = all_rdd.map(r => Row(r._1, r._2(0).toDouble, r._2(1).toDouble, r._2(2).toDouble, r._2(3).toDouble))
    //通过编程方式动态构造元数据
    val structType = StructType(Array(
      StructField("label", DoubleType, true),
      StructField("f1", DoubleType, true),
      StructField("f2", DoubleType, true),
      StructField("f3", DoubleType, true),
      StructField("f4", DoubleType, true)
    ))
    val sqlContext = new SQLContext(sc)

    //进行RDD到DataFrame的转换
    candidate_df = sqlContext.createDataFrame(candidate_rdd, structType)
  }

  def load_model2(): Array[Double] = {
    val model2 = LinearRegressionModel.load(path2)
    val weight2 = model2.weights.toArray
    for (w <- weight2) println("*****************************w2=" + w + "*****************************")
    weight2
  }

  def sim(weight: Array[Double], weight2_resorted: Array[Double]): Unit = {
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
    println("***********************temp1=" + temp1 + ";temp2=" + temp2 + "*****************************")
    println("***********************member=" + member + ";denominator=" + denominator + "*****************************")
    val sim = member / denominator
    println("***********************sim=" + sim + "*****************************")
  }

}
