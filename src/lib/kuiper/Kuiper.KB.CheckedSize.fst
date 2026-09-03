module Kuiper.KB.CheckedSize

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn mul
  (x : szp)
  (y : sz)
  norewrite
  requires emp
  returns z : sz
  ensures pure (SZ.v z == SZ.v x * SZ.v y /\
                SZ.fits (SZ.v x * SZ.v y))
{
  let maxu32 : sz = SZ.uint_to_t 4294967295;
  dguard (y <=^ (maxu32 /^ x));
  SZ.(x *^ y)
}

inline_for_extraction noextract
fn mulp
  (x y : szp)
  norewrite
  requires emp
  returns z : szp
  ensures pure (SZ.v z == SZ.v x * SZ.v y /\
                SZ.fits (SZ.v x * SZ.v y))
{
  let z = mul x y;
  z
}

inline_for_extraction noextract
fn mulp3
  (x y z : szp)
  norewrite
  requires emp
  returns r : szp
  ensures pure (SZ.v r == SZ.v x * SZ.v y * SZ.v z /\
                SZ.fits (SZ.v x * SZ.v y * SZ.v z))
{
  let xy = mulp x y;
  let xyz = mulp xy z;
  xyz
}

inline_for_extraction noextract
fn mulp4
  (w x y z : szp)
  norewrite
  requires emp
  returns r : szp
  ensures pure (SZ.v r == SZ.v w * SZ.v x * SZ.v y * SZ.v z /\
                SZ.fits (SZ.v w * SZ.v x * SZ.v y * SZ.v z))
{
  let wxy = mulp3 w x y;
  let wxyz = mulp wxy z;
  wxyz
}

inline_for_extraction noextract
fn mulp5
  (v w x y z : szp)
  norewrite
  requires emp
  returns r : szp
  ensures pure
    (SZ.v r == SZ.v v * SZ.v w * SZ.v x * SZ.v y * SZ.v z /\
     SZ.fits (SZ.v v * SZ.v w * SZ.v x * SZ.v y * SZ.v z))
{
  let vwxy = mulp4 v w x y;
  let vwxyz = mulp vwxy z;
  vwxyz
}

inline_for_extraction noextract
fn add
  (x y : sz)
  norewrite
  requires emp
  returns z : sz
  ensures pure (SZ.v z == SZ.v x + SZ.v y /\
                SZ.fits (SZ.v x + SZ.v y))
{
  let maxu32 : sz = SZ.uint_to_t 4294967295;
  dguard (x <=^ (maxu32 -^ y));
  SZ.(x +^ y)
}

inline_for_extraction noextract
fn addp
  (x : szp)
  (y : sz)
  norewrite
  requires emp
  returns z : szp
  ensures pure (SZ.v z == SZ.v x + SZ.v y /\
                SZ.fits (SZ.v x + SZ.v y))
{
  let z = add x y;
  z
}

inline_for_extraction let () = ()
