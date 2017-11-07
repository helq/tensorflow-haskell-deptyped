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
  mul,
  matMul,
  argMax,
  softmax,
  scalar,
  oneHot,
  oneHot_,
  reduceMean,
  softmaxCrossEntropyWithLogits,
  equal,
  truncatedNormal,
  relu,
  sub,
  cast
) where

import           GHC.TypeLits (Nat, Symbol, KnownNat, natVal, TypeError, ErrorMessage(Text, ShowType, (:<>:)), type (-))
import           Data.Proxy (Proxy(Proxy))
import           Data.Promotion.Prelude (type Length, type Reverse, type If, type (:||), type (:==))

import           Data.Vector.Sized (Vector, toList)
import           Data.Int (Int8, Int16, Int32, Int64)
import           Data.Word (Word8, Word16)
import           Data.Complex (Complex)
import           Data.ByteString (ByteString)

import qualified TensorFlow.Ops as TF (constant, add, matMul, placeholder, argMax, scalar,
                                       softmax, oneHot, reduceMean, softmaxCrossEntropyWithLogits,
                                       equal, truncatedNormal, vector, mul, relu, sub, cast)
import qualified TensorFlow.Types as TF (TensorType, Shape(Shape), type OneOf)
--import qualified TensorFlow.Tensor as TF (Tensor)
import           TensorFlow.Tensor (Value)
import           TensorFlow.Build (Build, MonadBuild)

import           TensorFlow.DepTyped.Base (KnownNatList(natListVal), ShapeProduct)
import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder, UnionPlaceholder)

constant :: forall (s :: [Nat]) (n :: Nat) a.
            (TF.TensorType a, ShapeProduct s ~ n, KnownNatList s)
         => Vector n a
         -> Tensor s '[] Build a
constant v = Tensor $ TF.constant shape (toList v)
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy s)

placeholder :: forall (name :: Symbol) (shape :: [Nat]) a m.
               (TF.TensorType a, KnownNatList shape, MonadBuild m)
            => m (Placeholder name shape a)
placeholder = Tensor <$> TF.placeholder shape
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)


type family BroadcastShapes (shape1::[Nat]) (shape2::[Nat]) :: [Nat] where
  BroadcastShapes shape shape = shape
  BroadcastShapes '[1] shape2 = shape2 -- this base cases are necessary to allow things like randomParam in mnist-deptyped example
  BroadcastShapes shape1 '[1] = shape1
  BroadcastShapes shape1 shape2 = Reverse (BroadcastShapes' (Reverse shape1) (Reverse shape2) shape1 shape2)

type family BroadcastShapes' (revshape1::[Nat]) (revshape2::[Nat]) (shape1::[Nat]) (shape2::[Nat]) :: [Nat] where
  BroadcastShapes' '[] '[] _ _ = '[]
  BroadcastShapes' '[] shape2 _ _ = shape2
  BroadcastShapes' shape1 '[] _ _ = shape1
  BroadcastShapes' (n:shape1) (m:shape2) origshape1 origshape2 =
    If (n:==1 :|| n:==m)
        (m : BroadcastShapes' shape1 shape2 origshape1 origshape2)
        (If (m:==1)
             (n : BroadcastShapes' shape1 shape2 origshape1 origshape2)
             (TypeError ('Text "Error: shapes " ':<>: 'ShowType origshape1
                            ':<>: 'Text " and " ':<>: 'ShowType origshape2
                            ':<>: 'Text " cannot be broadcast. For more info in broadcasting rules: "
                            ':<>: 'Text "https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html")))

add :: forall (shape1 :: [Nat]) (shape2 :: [Nat]) (phs1 :: [(Symbol, [Nat])]) (phs2 :: [(Symbol, [Nat])]) v1 v2 a.
       TF.OneOf '[(Complex Double), (Complex Float), ByteString, Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
    => Tensor shape1 phs1 v1 a -> Tensor shape2 phs2 v2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build a
add (Tensor t1) (Tensor t2) = Tensor (t1 `TF.add` t2)

sub :: forall (shape1 :: [Nat]) (shape2 :: [Nat]) (phs1 :: [(Symbol, [Nat])]) (phs2 :: [(Symbol, [Nat])]) v1 v2 a.
       TF.OneOf '[(Complex Double), (Complex Float), Int32, Int64, Word16, Double, Float] a
    => Tensor shape1 phs1 v1 a -> Tensor shape2 phs2 v2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build a
sub (Tensor t1) (Tensor t2) = Tensor (t1 `TF.sub` t2)

mul :: TF.OneOf '[(Complex Double), (Complex Float), Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
    => Tensor shape1 phs1 v1 a -> Tensor shape2 phs2 v2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build a
mul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.mul` t2)

matMul :: forall (shapeout::[Nat]) (n::Nat) a (i::Nat) (o::Nat) (p::[(Symbol,[Nat])]) (q::[(Symbol,[Nat])]) v'1 v'2.
          (TF.OneOf '[(Complex Double), (Complex Float), Int32, Word16, Double, Float] a,
           shapeout ~ '[i,o])
       => Tensor '[i,n] p v'1 a -> Tensor '[n,o] q v'2 a -> Tensor '[i,o] (UnionPlaceholder p q) Build a
matMul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.matMul` t2)


type family RemoveAxisFromShape (idx::Nat) (shape::[Nat]) :: [Nat] where
  RemoveAxisFromShape idx shape = RemoveAxisFromShape' idx shape idx shape

type family RemoveAxisFromShape' (idx::Nat) (shape::[Nat]) (idxorig::Nat) (shapeorig::[Nat]) :: [Nat] where
  RemoveAxisFromShape' _ '[]     idx shape = TypeError ('Text "Index " ':<>: 'ShowType idx ':<>:
                                                        'Text " is out of bounds of shape " ':<>: 'ShowType shape ':<>:
                                                        'Text ". Valid values for index [0.." ':<>:
                                                        'ShowType (Length shape) ':<>: 'Text "]" )
  RemoveAxisFromShape' 0 (_:shs) _ _ = shs
  RemoveAxisFromShape' n (sh:shs) idx shape = sh : RemoveAxisFromShape' (n-1) shs idx shape

argMax :: forall (n::Nat) (output_shape::[Nat]) (shape::[Nat]) (phs::[(Symbol,[Nat])]) t output_type v.
          (KnownNat n,
           output_shape ~ RemoveAxisFromShape n shape,
           TF.OneOf '[(Complex Double), (Complex Float), Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] t,
           TF.OneOf '[Int32, Int64] output_type)
       => Proxy n
       -> Tensor shape phs v t
       -> Tensor output_shape phs Build output_type
argMax p (Tensor t) = Tensor $ TF.argMax t (TF.scalar (fromInteger $ natVal p :: Int64))

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

reduceMean :: forall (shape::[Nat]) (phs::[(Symbol,[Nat])]) a v.
              TF.OneOf '[Double, Float, Complex Float, Complex Double] a
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
      => Tensor shape1 phs1 v'1 a -> Tensor shape2 phs2 v'2 a -> Tensor (BroadcastShapes shape1 shape2) (UnionPlaceholder phs1 phs2) Build Bool
equal (Tensor t1) (Tensor t2) = Tensor (t1 `TF.equal` t2)

truncatedNormal :: forall (shape::[Nat]) a m.
                   (MonadBuild m,
                    TF.OneOf '[Word16, Double, Float] a,
                    KnownNatList shape)
                => m (Tensor shape '[] Value a)
truncatedNormal = Tensor <$> TF.truncatedNormal (TF.vector shape_)
  where shape_ = fmap fromInteger $ natListVal (Proxy :: Proxy shape)

relu :: TF.OneOf '[Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] a
     => Tensor shape phs v a -> Tensor shape phs Build a
relu (Tensor t) = Tensor $ TF.relu t

cast :: (TF.TensorType srcT, TF.TensorType dstT)
     => Tensor shape phs v srcT -> Tensor shape phs Build dstT
cast (Tensor t) = Tensor $ TF.cast t
