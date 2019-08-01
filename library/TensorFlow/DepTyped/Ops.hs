-- Copyright 2017-2018 Elkin Cruz.
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
{-# LANGUAGE FlexibleInstances    #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module TensorFlow.DepTyped.Ops (
  constant,
  placeholder,
  add,
  mul,
  matMul,
  batchMatMul,
  argMax,
  softmax,
  scalar,
  oneHot,
  reduceMean,
  softmaxCrossEntropyWithLogits,
  equal,
  truncatedNormal,
  relu,
  sub,
  cast,
  square,
  reshape,
  sigmoid,
  shape
) where

import           GHC.TypeLits (Nat, Symbol)
import           Data.Kind (Type)

import           Data.Vector.Sized (Vector, toList)
import           Data.Int (Int8, Int16, Int32, Int64)
import           Data.Word (Word8, Word16)
import           Data.Complex (Complex)
import           Data.ByteString (ByteString)

import qualified TensorFlow.Ops as TF (
    constant, add, matMul, placeholder, argMax, scalar,
    softmax, oneHot, reduceMean, softmaxCrossEntropyWithLogits,
    equal, truncatedNormal, vector, mul, relu, sub, cast, abs, sign,
    neg, reshape, shape
  )
import qualified TensorFlow.GenOps.Core as TF (square, sigmoid, batchMatMul)
import qualified TensorFlow.Types as TF (TensorType, Shape(Shape), type OneOf)
--import qualified TensorFlow.Tensor as TF (Tensor)
import           TensorFlow.Tensor (Value)
import           TensorFlow.Build (Build, MonadBuild)

import           TensorFlow.DepTyped.Base (
    NatSing, KnownNat, KnownNats, NatList, Product, UnionPlaceholder,
    BroadcastShapes, RemoveAxisFromShape, type (++), Length, MatMulResult,
  )
import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder)
import           Data.Singletons (fromSing, sing)

-- This instance allows us to write "simpler" code when we want to create a constant vector with ease
-- e.g.,
-- do
--    w <- TFD.initializedVariable @'[1] 0
-- w has type "Tensor '[1] '[] Build a"
instance ( TF.TensorType a
         , Num a
         , v ~ Build
         , TF.OneOf '[ Double, Float, Int32, Int64
                     , Complex Float, Complex Double] a) => Num (Tensor s '[] v a) where
    (Tensor t1) + (Tensor t2) = Tensor (t1 `TF.add` t2)
    (Tensor t1) * (Tensor t2) = Tensor (t1 `TF.mul` t2)
    (Tensor t1) - (Tensor t2) = Tensor (t1 `TF.sub` t2)
    abs (Tensor t)    = Tensor (TF.abs t)
    fromInteger       = Tensor . TF.scalar . fromInteger
    signum (Tensor t) = Tensor (TF.sign t)
    negate (Tensor t) = Tensor (TF.neg t)

constant :: forall (s :: [Nat]) (n :: Nat) a.
            (TF.TensorType a, Product s ~ n, KnownNats s)
         => Vector n a
         -> Tensor s '[] Build a
constant v = Tensor $ TF.constant shape' (toList v)
  where shape' = TF.Shape . fmap fromIntegral $ fromSing (sing :: NatList s)

-- TODO(helq): change [Nat] for [Dim]
placeholder :: forall (name :: Symbol) (shape :: [Nat]) a m.
               (TF.TensorType a, KnownNats shape, MonadBuild m)
            => m (Placeholder name shape a)
placeholder = Tensor <$> TF.placeholder shape'
  where shape' = TF.Shape . fmap fromIntegral $ fromSing (sing :: NatList shape)

-- TODO(helq): change [Nat] for [Dim]
add :: forall (shape1 :: [Nat]) (shape2 :: [Nat]) (phs1 :: [(Symbol, [Nat], Type)]) (phs2 :: [(Symbol, [Nat], Type)]) v1 v2 a.
       TF.OneOf '[(Complex Double), (Complex Float), ByteString, Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
    => Tensor shape1 phs1 v1 a -> Tensor shape2 phs2 v2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build a
add (Tensor t1) (Tensor t2) = Tensor (t1 `TF.add` t2)

sub :: forall (shape1 :: [Nat]) (shape2 :: [Nat]) (phs1 :: [(Symbol, [Nat], Type)]) (phs2 :: [(Symbol, [Nat], Type)]) v1 v2 a.
       TF.OneOf '[(Complex Double), (Complex Float), Int32, Int64, Word16, Double, Float] a
    => Tensor shape1 phs1 v1 a -> Tensor shape2 phs2 v2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build a
sub (Tensor t1) (Tensor t2) = Tensor (t1 `TF.sub` t2)

mul :: TF.OneOf '[(Complex Double), (Complex Float), Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
    => Tensor shape1 phs1 v1 a -> Tensor shape2 phs2 v2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build a
mul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.mul` t2)

-- TODO(helq): change [Nat] for [Dim]
matMul :: forall (n :: Nat) (i :: Nat) (o :: Nat)
          (p :: [(Symbol, [Nat], Type)]) (q :: [(Symbol, [Nat], Type)])
          a v'1 v'2 .
          (TF.OneOf '[(Complex Double), (Complex Float), Int32, Word16, Double, Float] a)
       => Tensor '[n, i] p v'1 a
       -> Tensor '[i, o] q v'2 a
       -> Tensor '[n, o] (UnionPlaceholder p q) Build a
matMul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.matMul` t2)

batchMatMul :: forall (shapeLeft :: [Nat]) (shapeRight :: [Nat])
          (p :: [(Symbol, [Nat], Type)]) (q :: [(Symbol, [Nat], Type)])
          a v'1 v'2 .
          (TF.OneOf '[(Complex Double), (Complex Float), Int32, Word16, Double, Float] a)
       => Tensor shapeLeft p v'1 a
       -> Tensor shapeRight q v'2 a
       -> Tensor (MatMulResult shapeLeft shapeRight) (UnionPlaceholder p q) Build a
batchMatMul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.batchMatMul` t2)

-- TODO(helq): change [Nat] for [Dim]
argMax :: forall (n::Nat) (shapeInput::[Nat]) (phs::[(Symbol,[Nat],Type)]) t output_type v.
          (TF.OneOf '[(Complex Double), (Complex Float), Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] t,
           TF.OneOf '[Int32, Int64] output_type)
       => NatSing n
       -> Tensor shapeInput phs v t
       -> Tensor (RemoveAxisFromShape n shapeInput) phs Build output_type
argMax s (Tensor t) = Tensor $ TF.argMax t (TF.scalar (fromIntegral $ fromSing s :: Int64))

-- TODO(helq): change [Nat] for [Dim]
softmax :: (TF.OneOf '[Word16, Double, Float] t, KnownNat batchSize, KnownNat outs)
        => Tensor '[batchSize, outs] phs v t
        -> Tensor '[batchSize, outs] phs Build t
softmax (Tensor t) = Tensor $ TF.softmax t

-- TODO(helq): change [Nat] for [Dim]
scalar :: TF.TensorType a => a -> Tensor '[] '[] Build a
scalar = Tensor . TF.scalar

-- TODO(helq): change [Nat] for [Dim]
oneHot :: (TF.TensorType t,
           KnownNat n,
           TF.OneOf '[Int32, Int64, Word8] tI)
       => NatSing n
       -> Tensor '[1] '[] v'2 t
       -> Tensor '[1] '[] v'3 t
       -> Tensor shape phs v'1 tI
       -> Tensor (shape ++ '[n]) phs Build t
oneHot s (Tensor t1) (Tensor t2) (Tensor tinput) = Tensor $ TF.oneHot tinput (TF.scalar (fromIntegral $ fromSing s :: Int32)) t1 t2

-- TODO(helq): change [Nat] for [Dim]
reduceMean :: forall (shape::[Nat]) (phs::[(Symbol,[Nat],Type)]) a v.
              TF.OneOf '[Double, Float, Complex Float, Complex Double] a
           => Tensor shape phs v a
           -> Tensor '[1] phs Build a
reduceMean (Tensor t) = Tensor $ TF.reduceMean t

-- TODO(helq): change [Nat] for [Dim]
softmaxCrossEntropyWithLogits :: (TF.OneOf '[Word16, Double, Float] a)
                              => Tensor '[batchSize, n] phs1 v'1 a
                              -> Tensor '[batchSize, n] phs2 v'2 a
                              -> (Tensor '[batchSize] (UnionPlaceholder phs1 phs2) Build a,
                                  Tensor '[batchSize, n] (UnionPlaceholder phs1 phs2) Build a)
softmaxCrossEntropyWithLogits (Tensor feats) (Tensor labels) = (Tensor loss, Tensor backprop)
  where (loss, backprop) = TF.softmaxCrossEntropyWithLogits feats labels

-- TODO(helq): change [Nat] for [Dim]
equal :: (TF.OneOf '[(Complex Double), (Complex Float), Bool, ByteString, Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a)
      => Tensor shape1 phs1 v'1 a -> Tensor shape2 phs2 v'2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build Bool
equal (Tensor t1) (Tensor t2) = Tensor (t1 `TF.equal` t2)

-- TODO(helq): change [Nat] for [Dim]
truncatedNormal :: forall (shape::[Nat]) a m.
                   (MonadBuild m,
                    TF.OneOf '[Word16, Double, Float] a,
                    KnownNats shape)
                => m (Tensor shape '[] Value a)
truncatedNormal = Tensor <$> TF.truncatedNormal (TF.vector shape_)
  where shape_ = fmap fromIntegral $ fromSing (sing :: NatList shape)

-- TODO(helq): change [Nat] for [Dim]
relu :: TF.OneOf '[Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
     => Tensor shape phs v a -> Tensor shape phs Build a
relu (Tensor t) = Tensor $ TF.relu t

-- TODO(helq): change [Nat] for [Dim]
cast :: (TF.TensorType srcT, TF.TensorType dstT)
     => Tensor shape phs v srcT -> Tensor shape phs Build dstT
cast (Tensor t) = Tensor $ TF.cast t

-- TODO(helq): change [Nat] for [Dim]
square :: TF.OneOf '[(Complex Double), (Complex Float), Int32, Int64, Word16, Double, Float] a
    => Tensor shape phs v a -> Tensor shape phs Build a
square (Tensor t) = Tensor (TF.square t)

reshape :: forall t shape1 shape2 phs v.
           (TF.TensorType t, Product shape1 ~ Product shape2, KnownNats shape2)
        => Tensor shape1 phs v t
        -> Tensor shape2 phs Build t
reshape (Tensor t) = Tensor $ TF.reshape t $ TF.vector
  (map fromIntegral (fromSing (sing :: NatList shape2)) :: [Int64])

sigmoid :: forall (phs::[(Symbol,[Nat],Type)]) a v s.
           TF.OneOf '[Complex Double, Complex Float, Word16, Double, Float] a
        =>   Tensor s phs v a -> Tensor s phs Build a
sigmoid (Tensor t) = Tensor (TF.sigmoid t)

shape :: forall t shape phs v.
         (TF.TensorType t, KnownNats shape) -- , TF.OneOf '[Int32, Int64] out_type
       => Tensor shape phs v t
       -> Tensor '[Length shape] phs Build Int32
shape (Tensor t) = Tensor (TF.shape t)
