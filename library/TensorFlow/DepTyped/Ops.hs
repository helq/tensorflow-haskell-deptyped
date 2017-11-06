{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE UndecidableInstances #-}

module TensorFlow.DepTyped.Ops (
  constant,
  placeholder,
  add,
  matMul,
  argMax,
  softmax,
  scalar,
  oneHot,
  oneHot_,
  reduceMean,
  softmaxCrossEntropyWithLogits,
  equal,
  truncatedNormal
) where

import           GHC.TypeLits (Nat, Symbol, KnownNat, natVal, TypeError, ErrorMessage(Text, ShowType, (:<>:)), type (-))
import           Data.Proxy (Proxy(Proxy))
import           Data.Promotion.Prelude (type Length)

import           Data.Vector.Sized (Vector, toList)
import           Data.Int (Int8, Int16, Int32, Int64)
import           Data.Word (Word8, Word16)
import           Data.Complex (Complex)
import           Data.ByteString (ByteString)

import qualified TensorFlow.Ops as TF (constant, add, matMul, placeholder, argMax, scalar,
                                       softmax, oneHot, reduceMean, softmaxCrossEntropyWithLogits,
                                       equal, truncatedNormal, vector)
import qualified TensorFlow.Types as TF (TensorType, Shape(Shape), type OneOf)
--import qualified TensorFlow.Tensor as TF (Tensor)
import           TensorFlow.Tensor (Value)
import           TensorFlow.Build (Build, MonadBuild)

import           TensorFlow.DepTyped.Base (KnownNatList(natListVal), ShapeProduct)
import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder, UnionPlaceholder)

constant :: forall a (s :: [Nat]) (n :: Nat).
            (TF.TensorType a, ShapeProduct s ~ n, KnownNatList s)
         => Vector n a
         -> Tensor s '[] Build a
constant v = Tensor $ TF.constant shape (toList v)
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy s)

placeholder :: forall a (shape :: [Nat]) (name :: Symbol) (n :: Nat) m.
               (TF.TensorType a, ShapeProduct shape ~ n, KnownNatList shape, MonadBuild m)
            => m (Placeholder name shape a)
placeholder = Tensor <$> TF.placeholder shape
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)

add :: forall a (shape :: [Nat]) (phs1 :: [(Symbol, [Nat])]) (phs2 :: [(Symbol, [Nat])]) v1 v2.
       TF.OneOf '[(Complex Double), (Complex Float), ByteString, Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
       => Tensor shape phs1 v1 a -> Tensor shape phs2 v2 a -> Tensor shape (UnionPlaceholder phs1 phs2) Build a
add (Tensor t1) (Tensor t2) = Tensor (t1 `TF.add` t2)

matMul :: (TF.OneOf '[(Complex Double), (Complex Float), Int32, Word16, Double, Float] a)
       => Tensor '[i,n] p Build a -> Tensor '[n,o] q Build a -> Tensor '[i,o] (UnionPlaceholder p q) Build a
matMul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.matMul` t2)


type family RemoveAxisFromShape (idx::Nat) (shape::[Nat]) :: [Nat] where
  RemoveAxisFromShape idx shape = RemoveAxisFromShape' idx shape idx shape

type family RemoveAxisFromShape' (idx::Nat) (shape::[Nat]) (idxorig::Nat) (shapeorig::[Nat]) :: [Nat] where
  RemoveAxisFromShape' _ '[]     idx shape = TypeError ('Text "Index " ':<>: 'ShowType idx ':<>:
                                                        'Text " is out of bounds of shape " ':<>: 'ShowType shape ':<>:
                                                        'Text ". Valid values for index [" ':<>:
                                                        'ShowType 0 ':<>: 'Text ".." ':<>: 'ShowType (Length shape) ':<>: 'Text "]" )
  RemoveAxisFromShape' 0 (_:shs) _ _ = shs
  RemoveAxisFromShape' n (sh:shs) idx shape = sh : RemoveAxisFromShape' (n-1) shs idx shape

argMax :: (KnownNat n,
           output_shape ~ RemoveAxisFromShape n shape,
           TF.OneOf '[(Complex Double), (Complex Float), Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] t,
           TF.OneOf '[Int32, Int64] output_type)
       => Tensor shape phs v t
       -> Proxy n
       -> Tensor output_shape phs Build output_type
argMax (Tensor t) p = Tensor $ TF.argMax t (TF.scalar (fromInteger $ natVal p :: Int64))

softmax :: (TF.OneOf '[Word16, Double, Float] t, KnownNat batchSize, KnownNat outs)
        => Tensor '[batchSize, outs] phs v t
        -> Tensor '[batchSize, outs] phs Build t
softmax (Tensor t) = Tensor $ TF.softmax t

scalar :: TF.TensorType a => a -> Tensor '[1] '[] Build a
scalar = Tensor . TF.scalar

type family AddAxisToEndShape (shape::[Nat]) (axisSize::Nat) where
  AddAxisToEndShape '[] axis = '[axis]
  AddAxisToEndShape (s:shs) axis = s : AddAxisToEndShape shs axis

oneHot :: (TF.TensorType t,
           KnownNat n,
           TF.OneOf '[Int32, Int64, Word8] tI,
           output_shape ~ AddAxisToEndShape shape n)
       => Proxy n
       -> Tensor '[1] '[] v'2 t
       -> Tensor '[1] '[] v'3 t
       -> Tensor shape phs v'1 tI
       -> Tensor output_shape phs Build t
oneHot p (Tensor t1) (Tensor t2) (Tensor tinput) = Tensor $ TF.oneHot tinput (TF.scalar (fromInteger $ natVal p :: Int32)) t1 t2

-- TODO(helq): search how to use oneHot with both tensors and "numbers" entered by hand,
-- see usage in non-dependent-typed version of example tensorflow mnist
oneHot_ :: (TF.TensorType t,
           KnownNat n,
           TF.OneOf '[Int32, Int64, Word8] tI,
           output_shape ~ AddAxisToEndShape shape n)
        => Proxy n
        -> t
        -> t
        -> Tensor shape phs v'1 tI
        -> Tensor output_shape phs Build t
oneHot_ p t1 t2 = oneHot p (scalar t1) (scalar t2)

reduceMean :: TF.OneOf '[Double, Float, Complex Float, Complex Double] a
           => Tensor shape phs v a
           -> Tensor '[1] phs Build a
reduceMean (Tensor t) = Tensor $ TF.reduceMean t

softmaxCrossEntropyWithLogits :: (TF.OneOf '[Word16, Double, Float] a,
                                  addphs ~ UnionPlaceholder phs1 phs2)
                              => Tensor '[batchSize, n] phs1 v'1 a
                              -> Tensor '[batchSize, n] phs2 v'2 a
                              -> (Tensor '[batchSize] addphs Build a, Tensor '[batchSize, n] addphs Build a)
softmaxCrossEntropyWithLogits (Tensor feats) (Tensor labels) = (Tensor loss, Tensor backprop)
  where (loss, backprop) = TF.softmaxCrossEntropyWithLogits feats labels

equal :: (TF.OneOf '[(Complex Double), (Complex Float), Bool, ByteString, Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a)
      => Tensor shape phs1 v'1 a -> Tensor shape phs2 v'2 a -> Tensor shape (UnionPlaceholder phs1 phs2) Build Bool
equal (Tensor t1) (Tensor t2) = Tensor (t1 `TF.equal` t2)

truncatedNormal :: forall (shape::[Nat]) a m.
                   (MonadBuild m,
                    TF.OneOf '[Word16, Double, Float] a,
                    KnownNatList shape)
                => m (Tensor shape '[] Value a)
truncatedNormal = Tensor <$> TF.truncatedNormal (TF.vector shape_)
  where shape_ = fmap fromInteger $ natListVal (Proxy :: Proxy shape)
