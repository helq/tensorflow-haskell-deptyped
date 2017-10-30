{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE DataKinds             #-}

module Main (main) where

import           Data.Maybe (fromJust)
import           Data.Int (Int64)
import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, fromList)

import           TensorFlow.DepTyped

main1 :: IO (VN.Vector Int64)
main1 = runSession $ do
  let (elems1 :: Vector 4 Int64) = fromJust $ fromList [1,2,3,4]
      (constant1 :: Tensor '[2,2] '[] Build Int64) = constant elems1
  run constant1

main2 :: IO (VN.Vector Float)
main2 = runSession $ do
  let (elems1 :: Vector 12 Float) = fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
      (elems2 :: Vector 6 Float)  = fromJust $ fromList [5,6,7,8,9,10]
      (elems3 :: Vector 8 Float)  = fromJust $ fromList [11,12,13,14,15,16,17,18]
      (constant1  :: Tensor '[4,3] '[] Build Float) = constant elems1
      (constant2  :: Tensor '[3,2] '[] Build Float) = constant elems2
      (constant3  :: Tensor '[4,2] '[] Build Float) = constant elems3
      (multTensor :: Tensor '[4,2] '[] Build Float) = constant1 `matMul` constant2
      (addTensor  :: Tensor '[4,2] '[] Build Float) = multTensor `add` constant3
  run addTensor

--main3 :: IO (VN.Vector Float)
main3 :: IO (Vector 4 Float)
main3 = runSession $ do
  (a :: Placeholder "a" '[2,2] Float) <- placeholder
  (b :: Placeholder "b" '[2,2] Float) <- placeholder
  result <- render $ a `add` b
  let (inputA :: TensorData "a" [2,2] Float) = encodeTensorData . fromJust $ fromList [1,2,3,4]
      (inputB :: TensorData "b" [2,2] Float) = encodeTensorData . fromJust $ fromList [5,6,7,8]
  runWithFeeds (feed b inputB :~~ feed a inputA :~~ NilFeedList) result

-- Something that cannot be warranted is that having the same name placeholders
-- won't fuck everything up, you can very much use the same name and shape for
-- two different placeholders and the system will type check but it will fail
-- on runtime

fails :: IO (VN.Vector Float)
fails = runSession $ do
  (a :: Placeholder "a" '[2,2] Float) <- placeholder
  (b :: Placeholder "b" '[2,2] Float) <- placeholder
  (c :: Placeholder "a" '[2,2] Float) <- placeholder
  result <- render $ a `add` (b `add` c)
  let (inputA :: TensorData "a" [2,2] Float) = encodeTensorData . fromJust $ fromList [1,2,3,4]
      (inputB :: TensorData "b" [2,2] Float) = encodeTensorData . fromJust $ fromList [5,6,7,8]
  runWithFeeds (feed b inputB :~~ feed a inputA :~~ NilFeedList) result

main :: IO ()
main = do
  main1 >>= print
  main2 >>= print
  main3 >>= print
  fails >>= print
