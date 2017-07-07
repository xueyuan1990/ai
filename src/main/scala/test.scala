package com.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext

/**
  * Created by xueyuan on 2017/5/12.
  */
object test {
  var d: Double = 0.5
  var s: Array[String] = Array("s")
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
    this.d = 0.9
    s = Array("s2")
    //testFunction1
    val a = sc.parallelize(1 to 3, 3)
    val in1 = a
    val result1 = in1.mapPartitions(testFunction1)
    println("testFunction1: " + result1.collect().mkString)
    //testFunction2
    testFunction2()
    //testFunction3
    val size = a.collect.size
    val d_array = sc.parallelize(Array.fill(size)(d), 3)
    val in3 = a.zip(d_array)
    val result3 = in3.mapPartitions(testFunction3)
    println("testFunction3: " + result3.collect().mkString)
    //4
    val num1 = 1
    var num2 = 2
    num2 = 22
    val result4 = in1.map(r => (r, this.d, s(0),sc,num1,num2))
    println("testFunction4: " + result4.collect().mkString)
  }

  def testFunction1(iter: Iterator[(Int)]): Iterator[(Double, Double, String)] = {
    var res = List[(Double, Double, String)]()
    while (iter.hasNext) {
      val cur = iter.next;
      res ::= (cur, d, s(0))
    }
    res.iterator
  }

  def testFunction2(): Unit = {
    println("testFunction2: d=" + d + ";s=" + s(0))
  }

  def testFunction3(iter: Iterator[(Int, Double)]): Iterator[(Double, Double, String)] = {
    var res = List[(Double, Double, String)]()
    while (iter.hasNext) {
      val cur = iter.next;
      res ::= (cur._1, cur._2, s(0))
    }
    res.iterator
  }

}
