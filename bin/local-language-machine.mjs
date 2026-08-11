#!/usr/bin/env node
// npm entry point.
//
// The desktop shell is a window pointed at the localhost server, so the server on its own is the
// whole product — which is what makes an npm install a real alternative to the signed .dmg for
// anyone who already has Node. The workspace is the directory you run this in (server.ts reads
// process.cwd()), and PORT overrides the default.
//
// Kept as a wrapper rather than pointing `bin` straight at the built server: npm needs a shebang
// to link an executable on POSIX, and the build output is plain tsc emit with no shebang.
import "../dist/preview/server.js";
