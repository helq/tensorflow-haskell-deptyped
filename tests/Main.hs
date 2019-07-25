{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import System.Random (randomIO)
import Data.Int (Int32)
import Data.Vector.Sized (Vector, replicateM, fromTuple, index)

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


main :: IO ()
main = defaultMain
  [ testReshape
  , testShape
  , testSigmoid
  ]
