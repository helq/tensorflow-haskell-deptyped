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

{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeInType           #-}
{-# LANGUAGE NoStarIsType         #-}

module TensorFlow.DepTyped.Base (
  KnownNatList(natListVal),
  ShapeProduct,
  KnownNatListLength,
  AddPlaceholder,
  UnionPlaceholder,
  PlaceholderNotInList,
  SortPlaceholderList,
  BroadcastShapes,
  RemoveAxisFromShape,
  AddAxisToEndShape
) where

import           GHC.TypeLits (Nat, KnownNat, natVal, type (*), Symbol, TypeError, ErrorMessage(Text, ShowType,
                               (:<>:)), type (-), type (+))
import           Data.Proxy (Proxy(Proxy))
import           Data.Singletons.Prelude (type If, type (<), type (>), type (||), type (==), type Reverse)
import           Data.Singletons.Prelude.Foldable (type Length)
import           Data.Kind (Constraint, Type)

class KnownNatList (ns :: [Nat]) where
   natListVal :: proxy ns -> [Integer]
-- Base case
instance KnownNatList '[] where
  natListVal _ = []
-- Inductive step
instance (KnownNat n, KnownNatList ns) => KnownNatList (n ': ns) where
  natListVal _ = natVal (Proxy :: Proxy n) : natListVal (Proxy :: Proxy ns)

type family KnownNatListLength (s :: [Nat]) :: Nat where
  KnownNatListLength '[] = 0
  KnownNatListLength (_ ': s) = 1 + KnownNatListLength s

type family ShapeProduct (s :: [Nat]) :: Nat where
  ShapeProduct '[] = 1
  ShapeProduct (m ': s) = m * ShapeProduct s

-- TODO(helq): should this type be added?, it's superflous and doesn't add much more than some nice looking name, but it hides what the real type underneat is
--type Constant shape t = Tensor shape '[] Build t

type family AddPlaceholder (name :: Symbol) (shape :: [Nat]) (t :: Type) (placeholders :: [(Symbol, [Nat], Type)]) where
  AddPlaceholder n s t '[] = '[ '(n, s, t) ]
  AddPlaceholder n s t ('(n, s, t) ': phs) = '(n, s, t) ': phs
  AddPlaceholder n1 s1 t1 ('(n2, s2, t2) ': phs) =
                   If (n1 < n2)
                      ('(n1, s1, t1) ': '(n2, s2, t2) ': phs)
                  (If (n1 > n2)
                      ('(n2, s2, t2) ': AddPlaceholder n1 s1 t1 phs)
                  (If (t1 == t2)
                      (TypeError ('Text "The placeholder " ':<>: 'ShowType n1 ':<>:
                                  'Text " appears to have defined two different shapes " ':<>:
                                  'ShowType s1 ':<>: 'Text " and " ':<>: 'ShowType s2))
                      (TypeError ('Text "The placeholder " ':<>: 'ShowType n1 ':<>:
                                  'Text " appears to have defined two different types " ':<>:
                                  'ShowType t1 ':<>: 'Text " and " ':<>: 'ShowType t2))))

-- TODO(helq): improve UnionPlaceholder, it should work like merge sort and not like bubble surt
type family UnionPlaceholder (placeholders1 :: [(Symbol, [Nat], Type)]) (placeholders2 :: [(Symbol, [Nat], Type)]) where
  UnionPlaceholder '[] phs = phs
  UnionPlaceholder phs '[] = phs
  UnionPlaceholder ('(n1, s1, t1) ': phs1) phs2 = UnionPlaceholder phs1 (AddPlaceholder n1 s1 t1 phs2)

type family PlaceholderNotInList (name :: Symbol) (placeholders :: [(Symbol, [Nat], Type)]) :: Constraint where
  PlaceholderNotInList name phs = PlaceholderNotInList' name phs phs

type family PlaceholderNotInList' (name :: Symbol)
                                  (placeholders :: [(Symbol, [Nat], Type)])
                                  (placeholdersAll :: [(Symbol, [Nat], Type)]) :: Constraint where
  PlaceholderNotInList' _    '[] _ = () -- a valid Constraint, it could have been 'True ~ 'True or any other satisfiable constraint
  PlaceholderNotInList' name ('(name,     shape, t):phs) phsAll = TypeError ('Text "Placeholder " ':<>: 'ShowType name ':<>: 'Text " already in placeholders list " ':<>: 'ShowType phsAll)
  PlaceholderNotInList' name ('(othername,shape, t):phs) phsAll = PlaceholderNotInList' name phs phsAll

-- TODO(helq): investigate, what happens if we create a Symbol on the fly, at
-- runtime? does it also fails gracely with a type error message or segfaults?

type family SortPlaceholderList (placeholders :: [(Symbol, [Nat], Type)]) where
  SortPlaceholderList phs = SortPlaceholderList' phs '[]

-- TODO(helq): Improve sorting algorithm, this is "equivalent" to bubble sort (not a big problem with small placeholder lists)
type family SortPlaceholderList' (phs :: [(Symbol, [Nat], Type)])
                                 (phsSorted :: [(Symbol, [Nat], Type)]) where
  SortPlaceholderList' '[] phsSorted = phsSorted
  SortPlaceholderList' ('(n,s,t)':phs) phsSorted = SortPlaceholderList' phs (AddPlaceholder n s t phsSorted)

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
    If (n == 1 || n == m)
        (m : BroadcastShapes' shape1 shape2 origshape1 origshape2)
        (If (m == 1)
             (n : BroadcastShapes' shape1 shape2 origshape1 origshape2)
             (TypeError ('Text "Error: shapes " ':<>: 'ShowType origshape1
                            ':<>: 'Text " and " ':<>: 'ShowType origshape2
                            ':<>: 'Text " cannot be broadcast. For more info in broadcasting rules: "
                            ':<>: 'Text "https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html")))

type family RemoveAxisFromShape (idx::Nat) (shape::[Nat]) :: [Nat] where
  RemoveAxisFromShape idx shape = RemoveAxisFromShape' idx shape idx shape

type family RemoveAxisFromShape' (idx::Nat) (shape::[Nat]) (idxorig::Nat) (shapeorig::[Nat]) :: [Nat] where
  RemoveAxisFromShape' _ '[]     idx shape = TypeError ('Text "Index " ':<>: 'ShowType idx ':<>:
                                                        'Text " is out of bounds of shape " ':<>: 'ShowType shape ':<>:
                                                        'Text ". Valid values for index [0.." ':<>:
                                                        'ShowType (Length shape) ':<>: 'Text "]" )
  RemoveAxisFromShape' 0 (_:shs) _ _ = shs
  RemoveAxisFromShape' n (sh:shs) idx shape = sh : RemoveAxisFromShape' (n-1) shs idx shape

type family AddAxisToEndShape (shape::[Nat]) (axisSize::Nat) :: [Nat] where
  AddAxisToEndShape '[] axis = '[axis]
  AddAxisToEndShape (s:shs) axis = s : AddAxisToEndShape shs axis
