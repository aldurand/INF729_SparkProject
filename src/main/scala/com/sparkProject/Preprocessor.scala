package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{udf, lower, from_unixtime, datediff, round, concat_ws}
import org.apache.spark.sql.SaveMode


object Preprocessor {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /** AUTHOR : ALEXANDRE DURAND
      * MS BIG DATA @TELECOM PARISTECH
      * PROMO 2017-2018
      */


    /** 1 - CHARGEMENT DES DONNEES **/

      /**   1.a)  **/
    val df = spark.read
      .option("header", true)   // Utilise la première ligne pr nommer les colonnes
      .option("nullValue", "false")   // Remplace "false" par des nullValue
      .option("inferSchema", "true")  // try to infer the data type of each column
      .csv("./train_clean.csv")


    /**   1.b)  **/
    println("NB of Rows = " + df.count)
    println("NB of Columns = " + df.columns.length)

    /**   1.c)  **/
    df.show()

    /**   1.d)  **/
    df.printSchema  //imprime le nom des colonnes et leur type

    /**   1.e)  **/
    // Les valeurs de la colonne "project_id" sont de la forme "kkst122333344..." (donc pas castable en int)
    val df2 = df.withColumn("goal", $"goal".cast("int"))
                .withColumn("deadline", $"deadline".cast("int"))
                .withColumn("state_changed_at", $"state_changed_at".cast("int"))
                .withColumn("created_at", $"created_at".cast("int"))
                .withColumn("launched_at", $"launched_at".cast("int"))
                .withColumn("backers_count", $"backers_count".cast("int"))
                .withColumn("final_status", $"final_status".cast("int"))
// Verification:
    df2.printSchema



    /** 2 - CLEANING **/

    /**   2.a)  **/
    df2.select("goal", "backers_count", "final_status").describe().show()


    /**   2.b)  **/
    val columns_str = Seq("project_id", "name", "desc", "keywords", "disable_communication", "country", "currency")
    df2.select(columns_str.head, columns_str.tail: _*).show()
      // On voit que les valeurs sont toutes mélangées entre les colonnes.


    /**   2.c)  **/
    df2.groupBy("disable_communication").count.orderBy($"count".desc)show()
        // permet de voir que la très grande majorité des données sont des NULL pour cette colonne
        // Elle ne sert donc pas à grand chose, on peut la supprimer.
    val df3 = df2.drop("disable_communication")
    df3.printSchema()


    /**   2.d)  Suppression des "Fuites du Futur"**/
    val df4 = df3.drop("backers_count", "state_changed_at")
    df4.printSchema()



    /**   2.e)  **/

    /**  Udf_country :
    Si country=null prendre la valeur de currency, sinon laisser la valeur country actuelle.
    On veut les résultat dans une nouvelle colonne “country2”.
    **/
    def udf_country = udf{(country: String, currency: String) =>
      if (country == null)
        currency
      else
        country
    }

    /**  Udf_currency:
    Si currency.length != 3, currency prend la valeur null, sinon laisser la valeur currency actuelle.
    On veut les résultats dans une nouvelle colonne “currency2”
    **/
    def udf_currency = udf{(currency: String) =>
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    // To use the created UDF:
    val df5: DataFrame = df4.withColumn("country2", udf_country($"country", $"currency"))
                            .withColumn("currency2", udf_currency($"currency"))
                            .drop("country", "currency")



    // Pour aider notre algorithme, on souhaite qu'un même mot écrit en minuscules ou majuscules ne soit pas deux
    // "entités" différentes. On met tout en minuscules
    val df6: DataFrame = df5
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    df6.show(50)




    /**   2.f)  **/
    //Afficher le nombre d’éléments de chaque classe (colonne final_status)
     df6.groupBy("final_status").count.orderBy($"count".desc)show()


    /**   2.g)
    Conserver uniquement les lignes qui nous intéressent pour le modèle: final_status = 0 (Fail) ou 1 (Success).
    Les autres valeurs ne sont pas définies et on les enlève **/
    val df7 = df6.filter($"final_status".isin(0, 1))
    df7.groupBy("final_status").count.orderBy($"count".desc)show()






    /**
      * =========================================
      *
      * FIN DU TRAVAIL PERSONNEL ICI.
      *
      * LE RESTE EST EXTRAIT DE LA CORRECTION.
      *
      * =========================================
      *
      */




    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/


    // a) b) c) features à partir des timestamp
    val dfDurations: DataFrame = df7
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2")) // datediff requires a dateType
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600.0, 3)) // here timestamps are in seconds, there are 3600 seconds in one hour
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)
      .drop("created_at", "deadline", "launched_at")



    // d)
    val dfText= dfDurations
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))



    /** VALEUR NULLES **/

    val dfReady: DataFrame = dfText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1
      ))

    // vérifier l'équilibrage pour la classification
    dfReady.groupBy("final_status").count.orderBy($"count".desc).show



    dfReady.show(50)
    println("NB of rows at the end of preprocess = " + dfReady.count)


    /** WRITING DATAFRAME **/

    dfReady.write.mode(SaveMode.Overwrite)
      .parquet("./prepared_trainingset")

  }

}
