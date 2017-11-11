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

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE TypeInType          #-}

module TensorFlow.DepTyped.Types (
  TensorData(TensorData, unTensorData),
  encodeTensorData,
) where

import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, fromSized)

import           GHC.TypeLits (Nat, Symbol)
import           Data.Kind (Type)
import           Data.Proxy (Proxy(Proxy))

import qualified TensorFlow.Types as TF (TensorDataType, TensorData, TensorType, encodeTensorData, Shape(Shape))

import           TensorFlow.DepTyped.Base (KnownNatList(natListVal), ShapeProduct)

newtype TensorData (n :: Symbol) (s :: [Nat]) (a :: Type) = TensorData {unTensorData :: TF.TensorData a}

encodeTensorData :: forall a (name :: Symbol) (shape :: [Nat]) (n :: Nat).
                 (TF.TensorType a, ShapeProduct shape ~ n, KnownNatList shape, TF.TensorDataType VN.Vector a)
                 => Vector n a
                 -> TensorData name shape a
encodeTensorData v = TensorData (TF.encodeTensorData shape $ fromSized v :: TF.TensorData a)
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)
