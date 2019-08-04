{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}

module NN (testsNN) where

import System.Random (randomIO)
import Numeric.Natural (Natural)
import Control.Monad (forM_)
import Control.Monad.IO.Class
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit.Approx (assertApproxEqual)

import Data.Vector.Sized (Vector, replicateM, unsafeIndex)

import qualified TensorFlow.DepTyped as TF
import qualified TensorFlow.DepTyped.Base as TF
import qualified TensorFlow.DepTyped.NN as NN

import Data.Singletons (sing)
import Data.Singletons.TypeLits (KnownNat, withKnownNat)
import Data.Singletons.Prelude.Num ((%*), type (*))

type D1 = 2
type D2 = 3

helperKnownNats :: forall a n'. TF.NatSing n' -> ((KnownNat n', KnownNat (n' * D1 * D2)) => a) -> a
helperKnownNats n c = withKnownNat n $ withKnownNat (n %* sing @D1 %* sing @D2) c

helperBatch :: Natural -> IO (NN.Batch '[D1, D2] '[] TF.Build Float)
helperBatch batchSize = case TF.toSing batchSize of
  TF.SomeSing (n :: TF.NatSing n') ->
    helperKnownNats n $ do
      nums :: Vector (n' * D1 * D2) Float <- replicateM randomIO
      pure $ NN.Batch n $ TF.constant @[n', D1, D2] nums

testBatchedOp :: Test
testBatchedOp = testCase "Deptyped Batched Op" $ do
  let batchNum = 2
  batch1 <- helperBatch batchNum
  vector2 :: Vector (D1 * D2) Float <- replicateM randomIO
  let matrix2 = TF.constant @[D1, D2] vector2
  case batch1 of
    NN.Batch (n :: TF.NatSing n') (matrix1 :: TF.Tensor '[n', D1, D2] '[] TF.Build Float) ->
      helperKnownNats n $ TF.runSession $ do
        vector3 :: Vector (n' * D1 * D2) Float <- TF.run $ TF.add matrix1 matrix2
        vector1 :: Vector (n' * D1 * D2) Float <- TF.run matrix1
        -- can't really move these assers out of the context of pattern matching due to the dependency on `n` type
        liftIO $ forM_ [(m, i) | m <- [0..fromIntegral batchNum - 1], i <- [0..6 - 1]] $ \(m, i) -> do
          let
            index = m * 6 + i
            expected = unsafeIndex vector1 index + unsafeIndex vector2 i
            actual = unsafeIndex vector3 index
          assertApproxEqual "kek" 0.001 expected actual


testsNN :: [Test]
testsNN = [testBatchedOp]
