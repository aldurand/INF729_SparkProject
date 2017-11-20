package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /** AUTHOR : ALEXANDRE DURAND
      * MS BIG DATA @TELECOM PARISTECH
      * PROMO 2017-2018
      */


   /** CHARGER LE DATASET **/

   val df_preprocessed = spark.read.parquet("./prepared_trainingset")

    df_preprocessed.show(5)
    df_preprocessed.printSchema()
    df_preprocessed.groupBy("final_status").count.show()



    /** TF-IDF **/

    /** a) 1er stage: La première étape est séparer les textes en mots (ou tokens) avec un tokenizer.
      */
      //Tokenizer = process of taking text (such as a sentence) and breaking it into individual terms (usually words)
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    /** b) 2e stage: Retirer les stop words pour ne pas encombrer le modèle avec des mots qui ne véhiculent pas de sens.
      *              Créer le 2ème stage avec la classe StopWordsRemover.
      */
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    /** c) 3e stage: La partie TF de TF-IDF est faite avec la classe CountVectorizer.
      */
    val countVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("tokens_vectorized")
      //.setVocabSize(3)
      //.setMinDF(2)  //minimum number (or fraction if < 1.0) of documents a term must appear in to be included in the vocabulary

    /** d) 4e stage: Trouvez la partie IDF. On veut écrire l’output de cette étape dans une colonne “tfidf”. */
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")



    /** CONVERSION DONNEES CATEGORIELLES ===> DONNEES NUMERIQUES **/

    /** e) 5e stage: Convertir la variable catégorielle “country2” en données numérique.
      *              On veut les résultats dans une colonne "country_indexed".  */
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    /** f) 6e stage: Convertir la variable catégorielle “currency2” en données numérique.
      *              On veut les résultats dans une colonne "currency_indexed".  */
    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")




    /** VECTOR ASSEMBLER **/

    /** g) 7e stage: Assembler les features "tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"
      *              dans une seule colonne “features”.  */
    // Combines a given list of columns into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")



    /** MODEL **/

    /** h) 8e stage: Le modèle de classification, il s’agit d’une régression logistique
      *              que vous définirez de la façon suivante:  */
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0) //alpha=0 : L2 penalty / alpha=1 : L1 penalty / 0<alpha<1 : combination of L1 and L2
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3)) //label=0 si proba d'appartenir à classe 0 > 0.7 / sinon label=1
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/

    /** i) Enfin, créer le pipeline en assemblant les 8 stages définis précédemment, dans le bon ordre.  */
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf, countryIndexer, currencyIndexer, assembler, lr))


    /** TRAINING AND GRID-SEARCH **/

        // Splitter les données en Training Set et Test Set
    /** j) Créer un dataFrame nommé “training” et un autre nommé “test” à partir du dataFrame chargé initialement
      *    de façon à le séparer en training et test sets dans les proportions 90%, 10% respectivement.  */
    val Array(training, test) = df_preprocessed.randomSplit(Array(0.9, 0.1))


        //Entraînement du classifieur et réglage des hyper-paramètres de l’algorithme
    /** k) Préparer la grid-search pour satisfaire les conditions explicitées ci-dessus
      *    puis lancer la grid-search sur le dataset “training” préparé précédemment.  */
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVectorizer.minDF, Array[Double](55, 75, 95))
      .build()

    // On veut utiliser le f1-score pour comparer les modèles
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")  // La métrique "f1" n'est pas dispo en BinaryClassification, d'où l'utilisation de Multiclass

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7) // 70% of the data will be used for training and the remaining 30% for validation.


    // récupérer les meilleurs paramètres
    val model = trainValidationSplit.fit(training)


        // Tester le modèle obtenu sur les données test
    /** l) Appliquer le meilleur modèle trouvé avec la grid-search aux données test.
      *    Mettre les résultats dans le dataFrame df_WithPredictions.
      *    Afficher le f1-score du modèle sur les données de test.  */

    val df_WithPredictions = model.transform(test).select("features", "final_status", "predictions")

    val f1_score = evaluator.evaluate(df_WithPredictions)

    println("\n" + "The best model (found with the training set) is now applied on the test set."
      + "\n" + "The f1 score on test set is : " + f1_score + "\n")


    /** m) Afficher df_WithPredictions.groupBy("final_status","predictions").count.show()  */
    df_WithPredictions.groupBy("final_status","predictions").count.show()

  }
}
