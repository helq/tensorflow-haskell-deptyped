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

{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs                #-}
{-# OPTIONS_GHC -Wno-orphans #-}


module TensorFlow.DepTyped.NN (
  Batch(Batch),
  sigmoidCrossEntropyWithLogits
) where

import           Numeric.Natural (Natural)
import           GHC.TypeLits (Nat, Symbol)
import           Data.Kind (Type)

import qualified TensorFlow.Types as TF (TensorType, type OneOf)
import qualified TensorFlow.NN as TF (sigmoidCrossEntropyWithLogits)
--import qualified TensorFlow.Tensor as TF (Tensor)
import           TensorFlow.Tensor (Value)
import           TensorFlow.Build (MonadBuild)

import           TensorFlow.DepTyped.Base
import           TensorFlow.DepTyped.Tensor (Tensor(Tensor))


data Batch shape phs a t where
 Batch ::
  forall (n :: Nat) (shape :: [Nat]) phs a (t :: Type).
  (KnownNats shape) =>
  NatSing n ->
  Tensor (n : shape) phs a t ->
  Batch shape phs a t

sigmoidCrossEntropyWithLogits :: forall (phs1::[(Symbol,[Nat],Type)]) (phs2::[(Symbol,[Nat],Type)]) a s m.
           (MonadBuild m, TF.OneOf '[Float, Double] a, TF.TensorType a, Num a)
           => Tensor s phs1 Value a
           -> Tensor s phs2 Value a
           -> m (Tensor s (UnionPlaceholder phs1 phs2) Value a)
sigmoidCrossEntropyWithLogits (Tensor t1) (Tensor t2) = do
    t <- TF.sigmoidCrossEntropyWithLogits t1 t2
    return (Tensor t)
