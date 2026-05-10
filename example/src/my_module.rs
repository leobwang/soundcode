use http::{Request, Response, StatusCode};

pub fn response(req: Request<()>) -> http::Result<Response<()>> {
    match req.uri().path() {
        "/" => index(req),
        "/foo" => foo(req),
        "/bar" => bar(req),
        _ => not_found(req),
    }
}

fn index(_req: Request<()>) -> http::Result<Response<()>> {
    todo!();
}

fn foo(_req: Request<()>) -> http::Result<Response<()>> {
    Response::builder().status(StatusCode::OK).body(())
}

fn bar(_req: Request<()>) -> http::Result<Response<()>> {
    Response::builder().status(StatusCode::OK).body(())
}

fn not_found(_req: Request<()>) -> http::Result<Response<()>> {
    Response::builder().status(StatusCode::NOT_FOUND).body(())
}