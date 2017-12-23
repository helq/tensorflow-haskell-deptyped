-- Copyright 2017 Elkin Cruz.
-- Copyright 2017 The TensorFlow Authors.
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

{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}

import Control.Monad (replicateM_)
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Minimize as TF

import qualified TensorFlow.DepTyped as TFD
import           Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as VS (replicateM, map)

import           GHC.TypeLits (KnownNat)

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- VS.replicateM @100 randomIO
    let yData = VS.map (\x->x*3 + 8) xData
    -- Fit linear regression model.
    (w, b) <- fit xData yData
    assertBool "w == 3" (abs (3 - w) < 0.001)
    assertBool "b == 8" (abs (8 - b) < 0.001)
    putStrLn $ "w == " ++ show w
    putStrLn $ "b == " ++ show b

fit :: forall n. KnownNat n => Vector n Float -> Vector n Float -> IO (Float, Float)
fit xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TFD.constant @'[n] xData
        y = TFD.constant @'[n] yData
    -- Create scalar variables for slope and intercept.
    w <- TFD.initializedVariable @'[1] 0
    b <- TFD.initializedVariable @'[1] 0
    -- Define the loss function.
    let yHat = (x `TFD.mul` TFD.readValue w) `TFD.add` TFD.readValue b
        loss = TFD.square (yHat `TFD.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TFD.minimizeWith (TF.gradientDescent 0.001) loss [TFD.unVariable w, TFD.unVariable b]
    replicateM_ 1000 $ do
      () <- TFD.run trainStep -- this is necessary for haskell to select the right instance of `TFD.run`
      return ()               -- alternatively, you could annotate `replicateM_` with `Int -> IO () -> IO ()`
    -- Return the learned parameters.
    TF.Scalar w' <- TFD.run (TFD.readValue w)
    TF.Scalar b' <- TFD.run (TFD.readValue b)
    return (w', b')
