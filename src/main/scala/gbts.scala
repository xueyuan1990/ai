package com.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by xueyuan on 2017/6/14.
  */
object gbts {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null

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
    hiveContext = new HiveContext(sc)
    println("***********************hive*****************************")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    if ("reg".equals(args(0))) {
      training_reg()
    } else {
      training_clf()
    }


  }

  def training_clf(): Unit = {
    // Load and parse the data file.
    val data = load_data()
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    //    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    //    boostingStrategy.treeStrategy.numClasses = 2
    //    boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("***********************clf finished size=" + labelAndPreds.count() + "*****************************")
    for ((label, pre) <- labelAndPreds.take(10)) {
      println("*****************************(" + label + "," + pre + ")**********************************")
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
//    println("Learned classification GBT model:\n" + model.toDebugString)

    // Save and load model
    //    model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
    //    val sameModel = GradientBoostedTreesModel.load(sc,
    //      "target/tmp/myGradientBoostingClassificationModel")
  }

  def training_reg(): Unit = {
    val data = load_data()
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Regression use SquaredError by default.
    var boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(3) //Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.setMaxDepth(5)
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //      boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("***********************reg finished size=" + labelsAndPredictions.count() + "*****************************")
    for ((label, pre) <- labelsAndPredictions.take(10)) {
      println("*****************************(" + label + "," + pre + ")**********************************")
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression GBT model:\n" + model.toDebugString)

    // Save and load model
    //    model.save(sc, "target/tmp/myGradientBoostingRegressionModel")
    //    val sameModel = GradientBoostedTreesModel.load(sc,
    //      "target/tmp/myGradientBoostingRegressionModel")
  }

  def load_data(): RDD[LabeledPoint] = {
    val sql_1 = "select imei, sex, user_age, marriage_status, is_parent, mz_apps_car_owner, user_job, " +
      "user_life_city_lev, " +
      "app_contact_tag, app_education_tag, app_finance_tag, app_games_tag, app_health_tag, app_interact_tag, app_music_tag, " +
      "app_o2o_tag, app_read_tag, app_shopping_tag, app_travel_tag,  app_video_tag " +
      "from user_profile.idl_fdt_dw_tag limit 1000"
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    val data_rdd = df.map(r => (r.getString(1), r.getString(2), r.getString(3).toDouble, r.getString(4).toDouble, r.getString(5).toDouble, r.getString(6).toDouble, r.getString(7).toDouble))
    val lable_feature_rdd = data_rdd.map(r => {
      var sex = 0.0
      if ("male".equals(r._2)) {
        sex = 1
      } else if ("female".equals(r._2)) {
        sex = 2
      }
      val array = Array(sex, r._3, r._4, r._5, r._6, r._7)
      val dv = new DenseVector(array)
      val label = Math.random()
      if(label>0.5){
        (1,dv)
      }else{
        (0,dv)
      }

    })
    val label_array = lable_feature_rdd.map(r => r._1).take(100)
    for (i <- label_array) {
      print(i + ", ")
    }
    println()
    lable_feature_rdd.map(r => new LabeledPoint(r._1, r._2))
  }
}
