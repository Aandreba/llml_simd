use std::any::Any;
use std::borrow::Borrow;
use std::ops::Deref;

use proc_macro2::{Literal, TokenStream, Span};
use quote::__private::ext::RepToTokensExt;
use quote::{quote, ToTokens};
use syn::*;

#[proc_macro_attribute]
pub fn assign_targets (_input: proc_macro::TokenStream, alt: proc_macro::TokenStream) -> proc_macro::TokenStream {
    alt
}

#[proc_macro_attribute]
pub fn assign_rhs (_input: proc_macro::TokenStream, alt: proc_macro::TokenStream) -> proc_macro::TokenStream {
    alt
}

#[proc_macro_derive(Assign)]
pub fn assign_macro (input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let DeriveInput { 
        generics, 
        attrs, 
        vis: _, 
        ident, 
        data
    } = parse_macro_input!(input as DeriveInput);

    let targets = attrs.iter()
        .filter_map(|x| match x.parse_meta().unwrap() {
            Meta::List(list) => if list.path.is_ident("assign_targets") {
                return Some(list.nested.into_iter()
                    .map(|y| match y {
                        NestedMeta::Meta(meta) => meta.path().get_ident().unwrap().clone(),
                        _ => panic!("Unexpected error")
                    }))
            } else { None },
            _ => None
        })
        .flatten();

    let rhss = attrs.iter()
        .filter_map(|x| match x.parse_meta().unwrap() {
            Meta::List(list) => if list.path.is_ident("assign_rhs") {
                return Some(list.nested.into_iter()
                    .map(|y| match y {
                        NestedMeta::Meta(meta) => meta.path().get_ident().unwrap().clone(),
                        _ => panic!("Unexpected error")
                    }))
            } else { None },
            _ => None
        })
        .flatten()
        .collect::<Vec<Ident>>();

    let mut output = Vec::new();
    for target in targets {
        for rhs in rhss.iter() {
            output.push(assign_macro_impl(ident.clone(), generics.clone(), target.clone(), rhs.clone()));
        }
    }

    let output = quote! { #(#output)* };
    output.into()
}

fn assign_macro_impl (target: Ident, generics: Generics, original: Ident, rhs: Ident) -> proc_macro2::TokenStream {
    let original_name = format!("{original}");
    let original_fun = Ident::new(original_name.to_lowercase().as_str(), original.span());

    let assign_trait = Ident::new(&format!("{}Assign", original_name), original.span());
    let assign_fun = Ident::new(&format!("{}_assign", original_name.to_lowercase()), original.span());

    quote! {
        impl #assign_trait<#rhs> for #target #generics {
            #[inline(always)]
            fn #assign_fun (&mut self, rhs: #rhs) {
                *self = #original::<#rhs>::#original_fun(*self, rhs)
            } 
        } 
    }
}

// ARRAY GENERATOR
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::token::Comma;

struct ArrInput {
    expr: Expr,
    len: Lit
}

impl Parse for ArrInput {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let expr = input.parse::<Expr>()?;
        input.parse::<Token![;]>()?;
            
        Ok(ArrInput {
            expr,
            len: input.parse::<Lit>()?
        })
    }
}

#[proc_macro]
pub fn arr (input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ArrInput);

    let len = match input.len {
        Lit::Int(x) => x.base10_parse::<usize>().unwrap(),
        _ => panic!("Only integers are valid as array lengths")
    };

    let expresions = (0..len).into_iter()
        .map(|i| match input.expr.clone() {
            Expr::Lit(lit) => lit.into_token_stream(),
            Expr::Closure(c) => {
                assert!(c.inputs.len() == 1 && matches!(&c.inputs[0], Ident), "Invalid expresion");
                let ident = match &c.inputs[0] {
                    Pat::Ident(ident) => ident.clone().ident,
                    _ => panic!("Input is not an identity")
                };

                replace_ident(*c.body, ident, Lit::Int(Literal::usize_suffixed(i).into())).into_token_stream()
            },
            _ => panic!("Invalid array input")
        }).collect::<Punctuated<TokenStream, Comma>>();

    let output = quote! { [#expresions] };
    output.into()
}

fn replace_ident (expr: impl Into<Expr>, find: Ident, replace: Lit) -> Expr {
    match expr.into() {
        Expr::Array(mut array) => {
            array.elems = array.elems.into_iter()
                .map(|elem| replace_ident(elem, find.clone(), replace.clone()))
                .collect::<Punctuated<Expr, Comma>>();

            Expr::Array(array)
        },

        Expr::Assign(mut assign) => {
            assign.right = Box::new(replace_ident(*assign.right, find.clone(), replace.clone()));
            Expr::Assign(assign)
        },

        Expr::AssignOp(mut assign) => {
            assign.right = Box::new(replace_ident(*assign.right, find.clone(), replace.clone()));
            Expr::AssignOp(assign)
        },

        Expr::Binary(mut bin) => {
            bin.left = Box::new(replace_ident(*bin.left, find.clone(), replace.clone()));
            bin.right = Box::new(replace_ident(*bin.right, find.clone(), replace.clone()));
            Expr::Binary(bin)
        },

        Expr::Path(path) => {
            let true_path = path.clone().path;
            if true_path.segments.len() == 1 && true_path.segments[0].ident == find { return Expr::Lit(ExprLit { attrs: path.attrs, lit: replace }); }
            Expr::Path(path)
        },

        Expr::Unary(mut unary) => {
            unary.expr = Box::new(replace_ident(*unary.expr, find.clone(), replace.clone()));
            Expr::Unary(unary)
        },

        Expr::MethodCall(mut call) => {
            call.receiver = Box::new(replace_ident(*call.receiver, find.clone(), replace.clone()));
            call.args = call.args.into_iter()
                .map(|elem| replace_ident(elem, find.clone(), replace.clone()))
                .collect::<Punctuated<Expr, Comma>>();
            Expr::MethodCall(call)
        },

        Expr::Group(mut group) => {
            group.expr = Box::new(replace_ident(*group.expr, find.clone(), replace.clone()));
            Expr::Group(group)
        },

        Expr::Paren(mut paren) => {
            paren.expr = Box::new(replace_ident(*paren.expr, find.clone(), replace.clone()));
            Expr::Paren(paren)
        },

        Expr::Cast(mut cast) => {
            cast.expr = Box::new(replace_ident(*cast.expr, find.clone(), replace.clone()));
            Expr::Cast(cast)
        },

        Expr::Index(mut idx) => {
            idx.expr = Box::new(replace_ident(*idx.expr, find.clone(), replace.clone()));
            idx.index = Box::new(replace_ident(*idx.index, find.clone(), replace.clone()));
            Expr::Index(idx)
        },

        Expr::Field(mut field) => {
            field.base = Box::new(replace_ident(*field.base, find.clone(), replace.clone()));
            Expr::Field(field)
        },

        Expr::Lit(lit) => Expr::Lit(lit),
        expr => panic!("Unidentified expression: {expr:?}")
    }
}

// GENERIC CONSTANTS
struct ConstsInput {
    pre: Ident,
    len: Lit,
    ty: Ident
}

impl Parse for ConstsInput {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let pre = input.parse::<Ident>()?;
        input.parse::<Token![;]>()?;
        let len = input.parse::<Lit>()?;
        input.parse::<Token![as]>()?;

        Ok(ConstsInput {
            pre,
            len,
            ty: input.parse::<Ident>()?
        })
    }
}


#[proc_macro]
pub fn consts (input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ConstsInput);

    let ty = input.ty;
    let len = match input.len {
        Lit::Int(x) => x.base10_parse::<usize>().unwrap(),
        _ => panic!("Only integers are valid as consts lengths")
    };


    let consts : Punctuated<proc_macro2::TokenStream, Comma> = (0..len).into_iter()
        .map(|i| Ident::new(&format!("{:?}{i}", input.pre), Span::call_site()))
        .map(|cst| quote! { const #cst: #ty })
        .collect();

    consts.to_token_stream().into()
}