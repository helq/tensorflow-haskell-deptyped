-- Copyright 2017 Elkin Cruz.
-- Copyright 2017 James Bowen.
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

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds           #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}

module Main (main) where

import           Data.Maybe (fromJust)
import           Data.Int (Int64, Int32)
--import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, fromList)

import           Data.Proxy (Proxy(Proxy))

import           TensorFlow.DepTyped

--main1 :: IO (VN.Vector Int64)
main1 :: IO (Vector 4 Int64)
main1 = runSession $ do
  let (elems1 :: Vector 4 Int64) = fromJust $ fromList [1,2,3,4]
      (constant1 :: Tensor '[2,2] '[] Build Int64) = constant elems1
  run constant1

--main2 :: IO (VN.Vector Float)
main2 :: IO (Vector 8 Float)
main2 = runSession $ do
  let (elems1 :: Vector 12 Float) = fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
      (elems2 :: Vector 6 Float)  = fromJust $ fromList [5,6,7,8,9,10]
      (elems3 :: Vector 8 Float)  = fromJust $ fromList [11,12,13,14,15,16,17,18]
      (constant1  :: Tensor '[4,3] '[] Build Float) = constant elems1
      (constant2  :: Tensor '[3,2] '[] Build Float) = constant elems2
      (constant3  :: Tensor '[4,2] '[] Build Float) = constant elems3
      (multTensor :: Tensor '[4,2] '[] Build Float) = constant1 `matMul` constant2
      (addTensor  :: Tensor '[4,2] '[] Build Float) = multTensor `add` constant3
  run addTensor

--main3 :: IO (VN.Vector Float)
main3 :: IO (Vector 4 Float)
main3 = runSession $ do
  (a :: Placeholder "a" '[2,2] Float) <- placeholder
  (b :: Placeholder "b" '[2,2] Float) <- placeholder
  result <- render $ a `add` b
  let (inputA :: TensorData "a" [2,2] Float) = encodeTensorData . fromJust $ fromList [1,2,3,4]
      (inputB :: TensorData "b" [2,2] Float) = encodeTensorData . fromJust $ fromList [5,6,7,8]
  runWithFeeds (feed b inputB :~~ feed a inputA :~~ NilFeedList) result

-- Something that cannot be warranted is that having the same name placeholders
-- won't fuck everything up, you can very much use the same name and shape for
-- two different placeholders and the system will type check but it will fail
-- on runtime

--fails :: IO (VN.Vector Float)
fails :: IO (Vector 4 Float)
fails = runSession $ do
  (a :: Placeholder "a" '[2,2] Float) <- placeholder
  (b :: Placeholder "b" '[2,2] Float) <- placeholder
  (c :: Placeholder "a" '[2,2] Float) <- placeholder
  result <- render $ a `add` (b `add` c)
  let (inputA :: TensorData "a" [2,2] Float) = encodeTensorData . fromJust $ fromList [1,2,3,4]
      (inputB :: TensorData "b" [2,2] Float) = encodeTensorData . fromJust $ fromList [5,6,7,8]
  runWithFeeds (feed b inputB :~~ feed a inputA :~~ NilFeedList) result

main4 :: IO (Vector 4 Double, Vector 4 Double)
main4 = runSession $ do
  let elems = fromJust $ fromList [1,2,3,4]
      (constant1 :: Tensor '[1,4] '[] Build Double) = constant elems
      (constant2 :: Tensor '[2,2] '[] Build Double) = constant elems
  logits1 <- run $ softmax constant1
  logits2 <- run $ softmax constant2
  return (logits1, logits2)

main5 :: IO (Vector 20 Int32)
main5 = runSession $ do
  let elems = fromJust $ fromList [1,2,3,4]
      (constant1 :: Tensor '[2,2] '[] Build Int32) = constant elems
  run $ oneHot_ (Proxy :: Proxy 5) 1 0 constant1

main6 :: IO (Vector 1 Double)
main6 = runSession $ do
  let elems = fromJust $ fromList [3,4,3,7,4,4,4,6,3,13,15,0,4,2,6,0,6,8,12,15,12,11,5,0,11,14,10,13,12,11] -- from uniform distribution [0..15]
      (constant1 :: Tensor '[2,5,3] '[] Build Double) = constant elems
  run $ reduceMean constant1

main7 :: IO (Vector 4 Double)
main7 = runSession $ do
  let (logits :: Tensor '[4,3] '[] Build Double) = constant . fromJust $ fromList [14,4,12,1,13,13,4,2,9,10,0,5]
      (labels :: Tensor '[4,3] '[] Build Double) = constant . fromJust $ fromList [1,0,0,0,1,0,1,0,0,1,0,0]
  run $ fst $ softmaxCrossEntropyWithLogits logits labels

main8 :: IO (Vector 12 Double)
main8 = runSession $ do
  (randomValues :: Tensor '[4,3] '[] Value Double) <- truncatedNormal
  run randomValues

-- testing broadcast rules
main9 :: IO (Vector 24 Float, Vector 24 Float, Vector 12 Float)
main9 = runSession $ do
  let (constant1  :: Tensor   '[4,3] '[] Build Float) = constant . fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
      (constant2  :: Tensor '[2,1,3] '[] Build Float) = constant . fromJust $ fromList [1,-1,1,-1,1,-1]
      (mulTensor1 :: Tensor '[2,4,3] '[] Build Float) = constant1 `mul` constant2
  mulresult1 <- run mulTensor1
  let (constant3  :: Tensor '[1,1,4,3] '[] Build Float) = constant . fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
      (constant4  :: Tensor   '[2,1,3] '[] Build Float) = constant . fromJust $ fromList [1,-1,1,-1,1,-1]
      (mulTensor2 :: Tensor '[1,2,4,3] '[] Build Float) = constant3 `mul` constant4
  mulresult2 <- run mulTensor2
  let (constant5  :: Tensor '[2,2,3] '[] Build Float) = constant . fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
      (constant6  :: Tensor     '[1] '[] Build Float) = scalar 3.7
      (mulTensor3 :: Tensor '[2,2,3] '[] Build Float) = constant5 `mul` constant6
  mulresult3 <- run mulTensor3
  return (mulresult1, mulresult2, mulresult3)

main10 :: IO (Vector 8 Float)
main10 = runSession $ do
  (x :: Placeholder "x" '[4,3] Float) <- placeholder

  let elems1 = fromJust $ fromList [1,2,3,4,1,2]
      elems2 = fromJust $ fromList [5,6,7,8]
      (w :: Tensor '[3,2] '[] Build Float) = constant elems1
      (b :: Tensor '[4,1] '[] Build Float) = constant elems2
      y = (x `matMul` w) `add` b -- y shape: [4,2] (b shape is [4.1] but it broadcasts)

  let (inputX :: TensorData "x" [4,3] Float) = encodeTensorData . fromJust $ fromList [1,2,3,4,1,0,7,9,5,3,5,4]

  runWithFeeds (feed x inputX :~~ NilFeedList) y

main :: IO ()
main = do
  main1  >>= print
  main2  >>= print
  main3  >>= print
  main4  >>= print
  main5  >>= print
  main6  >>= print
  main7  >>= print
  main8  >>= print
  main9  >>= print
  main10 >>= print
  fails >>= print
