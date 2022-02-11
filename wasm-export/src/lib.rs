use core::ops::*;
use core::ptr::addr_of;
use wasm_bindgen::prelude::*;

macro_rules! wasm_import {
    (@self $target:ident, $($fun:ident),+) => {
        $(
            #[wasm_bindgen]
            impl $target {
                #[inline(always)]
                pub fn $fun (self) -> Self {
                    Self(self.0.$fun())
                }
            }
        )*
    };

    (@hoz $target:ident, $ty:ident, $($fun:ident),+) => {
        $(
            #[wasm_bindgen]
            impl $target {
                #[inline(always)]
                pub fn $fun (self) -> $ty {
                    self.0.$fun()
                }
            }
        )*
    };

    (@other $target:ident, $($fun:ident),+) => {
        $(
            #[wasm_bindgen]
            impl $target {
                #[inline(always)]
                pub fn $fun (self, other: $target) -> $target {
                    Self(self.0.$fun(other.0))
                }
            }
        )*
    };
}

macro_rules! wasm_export {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[wasm_bindgen]
            #[repr(transparent)]
            pub struct $target (wasm_export!(@tag $target, $ty));

            #[wasm_bindgen]
            impl $target {
                /// UNSAFE
                pub fn load (ptr: *const $ty) -> $target {
                    unsafe { Self(<wasm_export!(@tag $target, $ty)>::load(ptr)) }
                }

                #[wasm_bindgen(constructor)]
                pub fn new (a: &[$ty]) -> $target {
                    assert_eq!(a.len(), $len);
                    Self::load(a.as_ptr())
                }

                #[allow(non_snake_case)]
                #[inline(always)]
                pub fn filledWith (a: $ty) -> $target {
                    Self(<wasm_export!(@tag $target, $ty)>::filled_with(a))
                }

                #[inline(always)]
                pub fn get (&self, idx: usize) -> $ty {
                    unsafe { *(addr_of!(self.0) as *const $ty).add(idx) }
                }

                #[inline(always)]
                pub fn set (&mut self, idx: usize, value: $ty) {
                    self.0[idx] = value;
                }

                #[inline(always)]
                pub fn eq (&self, other: &$target) -> bool {
                    self.0.eq(&other.0)
                }

                #[inline(always)]
                pub fn ne (&self, other: &$target) -> bool {
                    self.0.ne(&other.0)
                }

                #[inline(always)]
                pub fn clone(&self) -> $target {
                    Self(self.0.clone())
                }

                #[allow(non_snake_case)]
                #[inline(always)]
                pub fn toArray(self) -> wasm_export!(@array $ty) {
                    let slice : &[$ty] = &Into::<[$ty;$len]>::into(self.0);
                    slice.into()
                }

                #[allow(non_snake_case)]
                #[inline(always)]
                pub fn toString(&self) -> js_sys::JsString {
                    dbg!("{self:?}").into()
                }
            }

            wasm_import!(
                @other $target, 
                add, sub, mul, div, vmin, vmax
            );

            wasm_import!(
                @hoz $target, $ty,
                min, max, sum
            );

            wasm_import!(
                @self $target,
                neg, abs, sqrt
            );
        )*
    };

    (@tag $target:ident, f32) => { llml_simd::float::single::$target };
    (@tag $target:ident, f64) => { llml_simd::float::double::$target };

    (@array f32) => { js_sys::Float32Array };
    (@array f64) => { js_sys::Float64Array };
}

wasm_export!(
    [f32;2] as f32x2,
    [f32;4] as f32x4,
    [f32;6] as f32x6,
    [f32;8] as f32x8,
    [f32;10] as f32x10,
    [f32;12] as f32x12,
    [f32;14] as f32x14,
    [f32;16] as f32x16,

    [f64;2] as f64x2,
    [f64;4] as f64x4,
    [f64;6] as f64x6,
    [f64;8] as f64x8,
    [f64;10] as f64x10,
    [f64;12] as f64x12,
    [f64;14] as f64x14,
    [f64;16] as f64x16
);