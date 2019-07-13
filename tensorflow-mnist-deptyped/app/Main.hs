-- Copyright 2017 Elkin Cruz.
-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE NoStarIsType        #-}

import Control.Monad (forM_, forM, when)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int32, Int64)
import Data.Word (Word8)
--import Data.List (genericLength)
import Data.List.Split (chunksOf)
import qualified Data.Text.IO as T
import Data.Maybe (fromJust)

import Data.Finite (finite)
import qualified Data.Vector.Sized as VS (Vector, concatMap, fromList, fromListN, index)
import Data.Monoid ((<>))
import Data.Maybe (fromMaybe)
import GHC.TypeLits (type (*), KnownNat, Nat)
--import GHC.TypeLits (someNatVal, SomeNat(SomeNat))
import Data.Proxy (Proxy(Proxy))

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Minimize as TF (adam)

import TensorFlow.Examples.MNIST.InputData (trainingImageData, testImageData, trainingLabelData, testLabelData)
import TensorFlow.Examples.MNISTDeptyped.Parse (readMNISTSamples, readMNISTLabels, drawMNIST, MNIST)

import qualified TensorFlow.DepTyped as TFD
import           TensorFlow.DepTyped (FeedList((:~~)))

numPixels :: Int64
numPixels = 28*28 :: Int64

type NumLabels = 10

-- | Create tensor with random values where the stddev depends on the width.
randomParam' :: forall (shape::[Nat]) m.
                (TFD.KnownNatList shape, TFD.MonadBuild m)
             => Int64 -> m (TFD.Tensor shape '[] TFD.Build Float)
randomParam' width =
    (`TFD.mul` stddev) <$> TFD.truncatedNormal @shape
  where
    stddev = TFD.scalar (1 / sqrt (fromIntegral width))

-- Types must match due to model structure.
--type LabelType = Int32

data ModelDep = ModelDep {
      train :: TFD.TensorData "images" '[100, 28*28] Float
            -> TFD.TensorData "labels" '[100] Int32
            -> TF.Session ()
    , infer :: TFD.TensorData "images" '[100, 28*28] Float
            -> TF.Session (VS.Vector 100 Int32)
    , errorRate :: TFD.TensorData "images" [100, 28*28] Float
                -> TFD.TensorData "labels" '[100] Int32
                -> TF.Session Float
    }

--TODO(helq): Make it possible to decide some shape sizes on runtime
--data ModelDep = ModelDep {
--      train :: forall n. KnownNat n
--            => TFD.TensorData "images" '[n, 28*28] Float
--            -> TFD.TensorData "labels" '[n] Int32
--            -> TF.Session ()
--    , infer :: forall n. KnownNat n
--            => TFD.TensorData "images" '[n, 28*28] Float
--            -> TF.Session (VS.Vector n Int32)
--    , errorRate :: forall n. KnownNat n
--                => TFD.TensorData "images" [n, 28*28] Float
--                -> TFD.TensorData "labels" '[n] Int32
--                -> TF.Session Float
--    }

createModel :: TF.Build ModelDep
createModel = do
  let numUnits = 500
  -- Inputs.
  images <- TFD.placeholder @"images" @'[100, 28*28]

  -- Hidden layer.
  hiddenWeights <- TFD.initializedVariable =<< randomParam' @'[28*28, 500] numPixels
  hiddenBiases  <- TFD.zeroInitializedVariable @'[500]
  -- TODO(helq): investigate why `mulHiddenZ` needs explicetely a type annotation, when it can alone deduce it
  -- A possible solution is to add another special case to BroadcastShapes (not very pretty)
  let mulHiddenZ = TFD.matMul @'[100, 500] images (TFD.readValue hiddenWeights)
      hiddenZ    = mulHiddenZ `TFD.add` (TFD.readValue hiddenBiases)
  let hidden = TFD.relu hiddenZ

  ---- Logits.
  logitWeights <- TFD.initializedVariable =<< randomParam' @'[500, NumLabels] numUnits
  logitBiases  <- TFD.zeroInitializedVariable @'[NumLabels]
  let logitsZ = TFD.matMul @[100, NumLabels] hidden (TFD.readValue logitWeights)
      logits  = logitsZ `TFD.add` TFD.readValue logitBiases
      prediction = TFD.argMax (Proxy :: Proxy 1) (TFD.softmax logits)
  predict <- TFD.render @_ @_ @Int32 @TF.Build prediction

  -- Create training action.
  labels <- TFD.placeholder @"labels" @'[100]
  let labelVecs = TFD.oneHot_ (Proxy :: Proxy NumLabels) 1 0 labels
      loss   = TFD.reduceMean $ fst $ TFD.softmaxCrossEntropyWithLogits logits labelVecs
      params = [TFD.unVariable hiddenWeights, -- this is the most unsecure part of all dependent typed tensorflow haskell example
                TFD.unVariable hiddenBiases,  -- TODO(helq): investigate how much more complexity is added if minimizeWith receives
                TFD.unVariable logitWeights,  --   a "list" of TFD.Variables, at the style of runWithFeeds
                TFD.unVariable logitBiases]
  trainStep <- TFD.minimizeWith TF.adam loss params

  let correctPredictions = TFD.equal predict labels
  errorRateTensor <- TFD.render $ TFD.scalar 1 `TFD.sub` TFD.reduceMean @_ @_ @Float (TFD.cast correctPredictions)

  return ModelDep {
        train = \imFeed lFeed -> TFD.runWithFeeds
                                 (TFD.feed images imFeed :~~ TFD.feed labels lFeed :~~ TFD.NilFeedList)
                                 trainStep
      , infer = \imFeed -> TFD.runWithFeeds (TFD.feed images imFeed :~~ TFD.NilFeedList) predict
      , errorRate = \imFeed lFeed -> TF.unScalar <$>
                                        TFD.runWithFeeds (
                                          TFD.feed images imFeed :~~
                                          TFD.feed labels lFeed  :~~
                                          TFD.NilFeedList
                                        ) errorRateTensor
      }

main :: IO ()
--main = do
--  let batchSize = 100
--  case someNatVal batchSize of
--    Nothing -> putStrLn $ show batchSize <> " is not a valid batch size"
--    Just (SomeNat (_ :: Proxy batchSize)) -> TF.runSession $ do
main = TF.runSession $ do
      let batchSize = 100 :: Integer

      trainingImages <- liftIO (readMNISTSamples =<< trainingImageData)
      trainingLabels <- liftIO (readMNISTLabels =<< trainingLabelData)
      testImages <- liftIO (readMNISTSamples =<< testImageData)
      testLabels <- liftIO (readMNISTLabels =<< testLabelData)

      -- Create the model.
      model <- TF.build createModel

      -- Functions for generating batches.
      let encodeImageBatch :: KnownNat bs => VS.Vector bs MNIST -> TFD.TensorData "images" '[bs, 28*28] Float
          encodeImageBatch xs = TFD.encodeTensorData $ fromIntegral <$> VS.concatMap id xs
          encodeLabelBatch :: KnownNat bs => VS.Vector bs Word8 -> TFD.TensorData "labels" '[bs] Int32
          encodeLabelBatch xs = TFD.encodeTensorData $ fromIntegral <$> xs

          chunkToVector :: (KnownNat n, Show a) => [a] -> VS.Vector n a
          chunkToVector chunk = fromMaybe (error $ "Error converting list " <> show chunk <> " into vector") $ VS.fromList chunk

      --let imageBatches = fmap chunkToVector . chunksOf (fromIntegral batchSize) $ cycle trainingImages :: [VS.Vector batchSize MNIST]
      --    labelBatches = fmap chunkToVector . chunksOf (fromIntegral batchSize) $ cycle trainingLabels :: [VS.Vector batchSize Word8]
      let imageBatches = fmap chunkToVector . chunksOf (fromIntegral batchSize) $ cycle trainingImages :: [VS.Vector 100 MNIST]
          labelBatches = fmap chunkToVector . chunksOf (fromIntegral batchSize) $ cycle trainingLabels :: [VS.Vector 100 Word8]

      -- TODO(helq): investigate why is this using so much memory, it shouldn't!!
      -- Train.
      forM_ (zip3 imageBatches labelBatches [0..(1000::Integer)]) $ \(imageBatch, labelBatch, i) -> do
          let images = encodeImageBatch imageBatch
              labels = encodeLabelBatch labelBatch

          train model images labels
          when (i `mod` 100 == 0) $ do
              err <- errorRate model images labels
              liftIO $ putStrLn $ "training error " ++ show (err * 100)
      liftIO $ putStrLn ""

      ---- Test.
      let imageTestBatches = fmap chunkToVector . chunksOf (fromIntegral batchSize) $ testImages :: [VS.Vector 100 MNIST]
          labelTestBatches = fmap chunkToVector . chunksOf (fromIntegral batchSize) $ testLabels :: [VS.Vector 100 Word8]
      testErrAcc <- forM (zip imageTestBatches labelTestBatches) $ \(ti, tl) ->
        errorRate model (encodeImageBatch ti) (encodeLabelBatch tl)
      let testErr = sum testErrAcc / 100 -- a total of 100 chunks because there are 10000 total images on testImages
      liftIO $ putStrLn $ "test error " ++ show (testErr * 100)

      -- Show some predictions.
      -- testPreds has size 100 because of the constraints on size :S
      testPreds <- infer model (encodeImageBatch . fromJust $ VS.fromListN testImages)
      liftIO $ forM_ ([0..3] :: [Int]) $ \i -> do
          putStrLn ""
          T.putStrLn $ drawMNIST $ testImages !! i
          putStrLn $ "expected " ++ show (testLabels !! i)
          let fi = finite $ fromIntegral i -- this is actually insecure, it can throw an error at runtime
          putStrLn $ "     got " ++ show (testPreds `VS.index` fi)

      return ()
