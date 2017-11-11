-- Copyright 2017 Elkin Cruz.
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

{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.DepTyped.Variable (
  Variable(Variable, unVariable),
  initializedVariable,
  zeroInitializedVariable,
  readValue
) where

import           GHC.TypeLits (Nat)

import qualified TensorFlow.Variable as TF (Variable, initializedVariable, zeroInitializedVariable,
                                            readValue)
import qualified TensorFlow.Types as TF (TensorType, Shape(Shape))
import qualified TensorFlow.Build as TF (MonadBuild)
import           TensorFlow.Build (Build)

import           TensorFlow.DepTyped.Tensor (Tensor(Tensor))
import           TensorFlow.DepTyped.Base (KnownNatList(natListVal))
import           Data.Proxy (Proxy(Proxy))

newtype Variable (shape :: [Nat]) a = Variable { unVariable :: TF.Variable a }

initializedVariable :: forall (shape::[Nat]) a v m.
                       (TF.TensorType a, TF.MonadBuild m)
                    => Tensor shape '[] v a -> m (Variable shape a)
initializedVariable (Tensor t) = Variable <$> TF.initializedVariable t

zeroInitializedVariable :: forall (shape :: [Nat]) a m.
                           (TF.MonadBuild m, TF.TensorType a, Num a, KnownNatList shape) => m (Variable shape a)
zeroInitializedVariable = Variable <$> TF.zeroInitializedVariable shape
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)

readValue :: TF.TensorType a => Variable shape a -> Tensor shape '[] Build a
readValue (Variable v) = Tensor $ TF.readValue v
