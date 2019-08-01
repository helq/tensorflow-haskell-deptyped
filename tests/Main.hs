{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad (forM_)
import System.Random (randomIO)
import Data.Int (Int32)
import Data.Vector.Sized (Vector, replicateM, fromTuple, index, empty, toList)

import Test.HUnit (assertEqual)
import Test.HUnit.Approx (assertApproxEqual)
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)

import qualified TensorFlow.DepTyped as TF

testReshape :: Test
testReshape = testCase "DepTyped Op Reshape" $ do
  vector <- replicateM randomIO :: IO (Vector 100 Float)
  result :: Vector 100 Float <- TF.runSession $ do
    let tensor = TF.constant vector :: TF.Tensor '[100] '[] TF.Build Float
    let reshaped = TF.reshape tensor :: TF.Tensor '[10, 10] '[] TF.Build Float
    TF.run reshaped
  assertEqual "Vectors are equal" vector result


testShape :: Test
testShape = testCase "DepTyped Op Shape" $ do
  vector <- replicateM randomIO :: IO (Vector 60 Float)
  shape :: Vector 3 Int32 <- TF.runSession $ do
    let tensor = TF.constant vector :: TF.Tensor '[3, 4, 5] '[] TF.Build Float
    TF.run $ TF.shape tensor
  assertEqual "Shape is correct" (fromTuple (3, 4, 5)) shape


testSigmoid :: Test
testSigmoid = testCase "DepTyped Op Sigmoid" $ do
  let vector = fromTuple (-1000, -1, 0, 1, 1000)
  result :: Vector 5 Float <- TF.runSession $ do
    let tensor = TF.constant vector :: TF.Tensor '[5] '[] TF.Build Float
    TF.run $ TF.sigmoid tensor
  assertApproxEqual "sigmoid(-1000)" 0.0001 0          $ index result 0
  assertApproxEqual "sigmoid(-1)"    0.0001 0.26894143 $ index result 1
  assertApproxEqual "sigmoid(0)"     0.0001 0.5        $ index result 2
  assertApproxEqual "sigmoid(1)"     0.0001 0.73105857 $ index result 3
  assertApproxEqual "sigmoid(1000)"  0.0001 1.0        $ index result 4

helperMatMul :: Int -> Int -> Int -> Int -> [Float] -> [Float] -> [Float] -> IO ()
helperMatMul batchSize is ks js vector1 vector2 result =
  forM_ [(m, i, j) | m <- [0..batchSize - 1], i <- [0..is - 1], j <- [0..js - 1]] $ \(m, i, j) -> do
    let l1 = [vector1 !! (m * is * ks + i * ks + k) | k <- [0..ks - 1]]
    let l2 = [vector2 !! (m * ks * js + k * js + j) | k <- [0..ks - 1]]
    let expected = sum $ zipWith (*) l1 l2
    assertApproxEqual "result is correct" 0.0001 expected $ result !! (m * is * js + i * js + j)

testMatMul :: Test
testMatMul = testCase "DepTyped Op MatMul" $ do
  vector1 <- replicateM randomIO :: IO (Vector 6 Float)
  vector2 <- replicateM randomIO :: IO (Vector 12 Float)
  result :: Vector 8 Float <- TF.runSession $ do
    let tensor1 = TF.constant vector1 :: TF.Tensor '[2, 3] '[] TF.Build Float
    let tensor2 = TF.constant vector2 :: TF.Tensor '[3, 4] '[] TF.Build Float
    TF.run $ TF.matMul tensor1 tensor2
  helperMatMul 1 2 3 4 (toList vector1) (toList vector2) (toList result)

testBatchMatMul :: Test
testBatchMatMul = testCase "DepTyped Op BatchMatMul" $ do
  vector1 <- replicateM randomIO :: IO (Vector 12 Float)
  vector2 <- replicateM randomIO :: IO (Vector 24 Float)
  result :: Vector 16 Float <- TF.runSession $ do
    let tensor1 = TF.constant vector1 :: TF.Tensor '[1, 2, 2, 3] '[] TF.Build Float
    let tensor2 = TF.constant vector2 :: TF.Tensor '[1, 2, 3, 4] '[] TF.Build Float
    TF.run $ TF.batchMatMul tensor1 tensor2
  helperMatMul 2 2 3 4 (toList vector1) (toList vector2) (toList result)

testScalar :: Test
testScalar = testCase "DepTyped Op Scalar" $ do
  let num = 1000
  shape <- TF.runSession $ do
    let scalar = TF.scalar num :: TF.Tensor '[] '[] TF.Build Float
    let shape = TF.shape scalar
    TF.run shape
  assertEqual "Shape is correct" empty shape


main :: IO ()
main = defaultMain
  [ testReshape
  , testShape
  , testSigmoid
  , testMatMul
  , testBatchMatMul
  , testScalar
  ]
