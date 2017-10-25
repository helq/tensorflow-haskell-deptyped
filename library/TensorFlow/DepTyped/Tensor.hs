{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstraintKinds       #-}

module TensorFlow.DepTyped.Tensor (
  KnownNatList(natListVal),
  AddPlaceholder,
  UnionPlaceholder,
  PlaceholderNotInList,
  SortPlaceholderList,

  Tensor(Tensor),
  Placeholder,
  Feed(Feed),
  FeedList(NilFeedList,(:~~)),
  render,
  feed
) where

import           GHC.TypeLits (Nat, Symbol, TypeError, ErrorMessage(Text, ShowType, (:<>:)))
import           Data.Singletons.Prelude (If, type (:<), type (:>))
import           Data.Kind (Constraint)

import           TensorFlow.Core (Build, Value, MonadBuild)
import qualified TensorFlow.Tensor as TF (Feed, feed, Tensor, render)
import qualified TensorFlow.Types as TF (TensorType)

import           TensorFlow.DepTyped.Types (TensorData(TensorData))
import           TensorFlow.DepTyped.Base (KnownNatList(natListVal))

data Tensor (s :: [Nat]) (p :: [(Symbol, [Nat])]) v a where
  Tensor :: (TF.TensorType a) => TF.Tensor v a -> Tensor s p v a

type Placeholder name shape t = Tensor shape '[ '(name, shape) ] Value t

-- TODO(helq): should this type be added?, it's superflous and doesn't add much more than some nice looking name, but it hides what the real type underneat is
--type Constant shape t = Tensor shape '[] Build t

type family AddPlaceholder (name :: Symbol) (shape :: [Nat]) (placeholders :: [(Symbol, [Nat])]) where
  AddPlaceholder n s '[] = '[ '(n, s) ]
  AddPlaceholder n s ('(n, s) ': phs) = '(n, s) ': phs
  AddPlaceholder n1 s1 ('(n2, s2) ': phs) =
                   If (n1 :< n2)
                      ('(n1, s1) ': '(n2, s2) ': phs)
                      (If (n1 :> n2)
                          ('(n2, s2) ': AddPlaceholder n1 s1 phs)
                          (TypeError ('Text "The placeholder " ':<>: 'ShowType n1 ':<>: 'Text " appears to have defined two different shapes " ':<>: 'ShowType s1 ':<>: 'Text " and " ':<>: 'ShowType s2)))

-- TODO(helq): improve UnionPlaceholder, it should work like merge sort and not like bubble surt
type family UnionPlaceholder (placeholders1 :: [(Symbol, [Nat])]) (placeholders2 :: [(Symbol, [Nat])]) where
  UnionPlaceholder '[] phs = phs
  UnionPlaceholder ('(n1, s1) ': phs1) phs2 = UnionPlaceholder phs1 (AddPlaceholder n1 s1 phs2)

newtype Feed (name :: Symbol) (shape :: [Nat]) a = Feed TF.Feed

type family PlaceholderNotInList (name :: Symbol) (placeholders :: [(Symbol, [Nat])]) :: Constraint where
  PlaceholderNotInList name phs = PlaceholderNotInList' name phs phs

type family PlaceholderNotInList' (name :: Symbol) (placeholders :: [(Symbol, [Nat])]) (placeholdersAll :: [(Symbol, [Nat])]) :: Constraint where
  PlaceholderNotInList' name '[] phsAll = () -- a valid Constraint, it could have been 'True ~ 'True or any other satisfiable constraint
  PlaceholderNotInList' name ('(name,     shape):phs) phsAll = TypeError ('Text "Placeholder " ':<>: 'ShowType name ':<>: 'Text " already in placeholders list " ':<>: 'ShowType phsAll)
  PlaceholderNotInList' name ('(othername,shape):phs) phsAll = PlaceholderNotInList name phs

data FeedList (placeholders :: [(Symbol, [Nat])]) a where
  NilFeedList :: FeedList '[] a
  (:~~) :: PlaceholderNotInList name phs
        => Feed name shape a
        -> FeedList phs a
        -> FeedList ('(name, shape) ': phs) a

infixr 5 :~~

-- TODO(helq): investigate, what happens if we create a Symbol on the fly, at
-- runtime? does it also fails gracely with a type error message or segfaults?

type family SortPlaceholderList (placeholders :: [(Symbol, [Nat])]) where
  SortPlaceholderList phs = SortPlaceholderList' phs '[]

-- TODO(helq): Improve sorting algorithm, this is "equivalent" to bubble sort (not a big problem with small placeholder lists)
type family SortPlaceholderList' (phs :: [(Symbol, [Nat])]) (phsSorted :: [(Symbol, [Nat])]) where
  SortPlaceholderList' '[] phsSorted = phsSorted
  SortPlaceholderList' ('(n,s)':phs) phsSorted = SortPlaceholderList' phs (AddPlaceholder n s phsSorted)

render :: MonadBuild m => Tensor shape plholders Build t -> m (Tensor shape plholders Value t)
render (Tensor t) = Tensor <$> TF.render t

-- TODO(helq): replace Placeholder for something more general as it used in the non-deptyped `feed`
feed :: Placeholder name shape a -> TensorData name shape a -> Feed name shape a
feed (Tensor t) (TensorData td) = Feed $ TF.feed t td
