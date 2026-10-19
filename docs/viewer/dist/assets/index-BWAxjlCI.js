(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))n(a);new MutationObserver(a=>{for(const o of a)if(o.type==="childList")for(const l of o.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&n(l)}).observe(document,{childList:!0,subtree:!0});function t(a){const o={};return a.integrity&&(o.integrity=a.integrity),a.referrerPolicy&&(o.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?o.credentials="include":a.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function n(a){if(a.ep)return;a.ep=!0;const o=t(a);fetch(a.href,o)}})();/**
 * @license
 * Copyright 2010-2025 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const sc="181",xr={ROTATE:0,DOLLY:1,PAN:2},fr={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},a0=0,hl=1,o0=2,Su=1,c0=2,ti=3,Ei=0,_n=1,Hn=2,ai=0,gr=1,pl=2,ml=3,_l=4,l0=5,Vi=100,u0=101,f0=102,d0=103,h0=104,p0=200,m0=201,_0=202,x0=203,po=204,mo=205,g0=206,v0=207,y0=208,S0=209,b0=210,E0=211,M0=212,T0=213,w0=214,_o=0,xo=1,go=2,br=3,vo=4,yo=5,So=6,bo=7,bu=0,A0=1,R0=2,bi=0,C0=1,P0=2,D0=3,L0=4,F0=5,I0=6,U0=7,Eu=300,Er=301,Mr=302,Eo=303,Mo=304,ua=306,To=1e3,ni=1001,wo=1002,vn=1003,N0=1004,Ts=1005,wn=1006,Ia=1007,Hi=1008,$n=1009,Mu=1010,Tu=1011,jr=1012,ac=1013,Wi=1014,ii=1015,Rr=1016,oc=1017,cc=1018,Yr=1020,wu=35902,Au=35899,Ru=1021,Cu=1022,Nn=1023,qr=1026,Kr=1027,Pu=1028,lc=1029,uc=1030,fc=1031,dc=1033,Ks=33776,Zs=33777,Js=33778,Qs=33779,Ao=35840,Ro=35841,Co=35842,Po=35843,Do=36196,Lo=37492,Fo=37496,Io=37808,Uo=37809,No=37810,Oo=37811,ko=37812,Bo=37813,zo=37814,Vo=37815,Go=37816,Ho=37817,Wo=37818,Xo=37819,$o=37820,jo=37821,Yo=36492,qo=36494,Ko=36495,Zo=36283,Jo=36284,Qo=36285,ec=36286,O0=3200,k0=3201,Du=0,B0=1,yi="",Mn="srgb",Tr="srgb-linear",ia="linear",Ft="srgb",Zi=7680,xl=519,z0=512,V0=513,G0=514,Lu=515,H0=516,W0=517,X0=518,$0=519,gl=35044,j0=35048,vl="300 es",Wn=2e3,ra=2001;function Fu(r){for(let e=r.length-1;e>=0;--e)if(r[e]>=65535)return!0;return!1}function sa(r){return document.createElementNS("http://www.w3.org/1999/xhtml",r)}function Y0(){const r=sa("canvas");return r.style.display="block",r}const yl={};function Sl(...r){const e="THREE."+r.shift();console.log(e,...r)}function Ze(...r){const e="THREE."+r.shift();console.warn(e,...r)}function Gt(...r){const e="THREE."+r.shift();console.error(e,...r)}function Zr(...r){const e=r.join(" ");e in yl||(yl[e]=!0,Ze(...r))}function q0(r,e,t){return new Promise(function(n,a){function o(){switch(r.clientWaitSync(e,r.SYNC_FLUSH_COMMANDS_BIT,0)){case r.WAIT_FAILED:a();break;case r.TIMEOUT_EXPIRED:setTimeout(o,t);break;default:n()}}setTimeout(o,t)})}class ji{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const a=n[e];if(a!==void 0){const o=a.indexOf(t);o!==-1&&a.splice(o,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const a=n.slice(0);for(let o=0,l=a.length;o<l;o++)a[o].call(this,e);e.target=null}}}const sn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],ea=Math.PI/180,tc=180/Math.PI;function ts(){const r=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(sn[r&255]+sn[r>>8&255]+sn[r>>16&255]+sn[r>>24&255]+"-"+sn[e&255]+sn[e>>8&255]+"-"+sn[e>>16&15|64]+sn[e>>24&255]+"-"+sn[t&63|128]+sn[t>>8&255]+"-"+sn[t>>16&255]+sn[t>>24&255]+sn[n&255]+sn[n>>8&255]+sn[n>>16&255]+sn[n>>24&255]).toLowerCase()}function dt(r,e,t){return Math.max(e,Math.min(t,r))}function K0(r,e){return(r%e+e)%e}function Ua(r,e,t){return(1-t)*r+t*e}function kr(r,e){switch(e.constructor){case Float32Array:return r;case Uint32Array:return r/4294967295;case Uint16Array:return r/65535;case Uint8Array:return r/255;case Int32Array:return Math.max(r/2147483647,-1);case Int16Array:return Math.max(r/32767,-1);case Int8Array:return Math.max(r/127,-1);default:throw new Error("Invalid component type.")}}function hn(r,e){switch(e.constructor){case Float32Array:return r;case Uint32Array:return Math.round(r*4294967295);case Uint16Array:return Math.round(r*65535);case Uint8Array:return Math.round(r*255);case Int32Array:return Math.round(r*2147483647);case Int16Array:return Math.round(r*32767);case Int8Array:return Math.round(r*127);default:throw new Error("Invalid component type.")}}const Z0={DEG2RAD:ea};class et{constructor(e=0,t=0){et.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,a=e.elements;return this.x=a[0]*t+a[3]*n+a[6],this.y=a[1]*t+a[4]*n+a[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=dt(this.x,e.x,t.x),this.y=dt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=dt(this.x,e,t),this.y=dt(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(dt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(dt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),a=Math.sin(t),o=this.x-e.x,l=this.y-e.y;return this.x=o*n-l*a+e.x,this.y=o*a+l*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Xi{constructor(e=0,t=0,n=0,a=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=a}static slerpFlat(e,t,n,a,o,l,f){let p=n[a+0],h=n[a+1],_=n[a+2],g=n[a+3],v=o[l+0],b=o[l+1],R=o[l+2],P=o[l+3];if(f<=0){e[t+0]=p,e[t+1]=h,e[t+2]=_,e[t+3]=g;return}if(f>=1){e[t+0]=v,e[t+1]=b,e[t+2]=R,e[t+3]=P;return}if(g!==P||p!==v||h!==b||_!==R){let E=p*v+h*b+_*R+g*P;E<0&&(v=-v,b=-b,R=-R,P=-P,E=-E);let m=1-f;if(E<.9995){const U=Math.acos(E),L=Math.sin(U);m=Math.sin(m*U)/L,f=Math.sin(f*U)/L,p=p*m+v*f,h=h*m+b*f,_=_*m+R*f,g=g*m+P*f}else{p=p*m+v*f,h=h*m+b*f,_=_*m+R*f,g=g*m+P*f;const U=1/Math.sqrt(p*p+h*h+_*_+g*g);p*=U,h*=U,_*=U,g*=U}}e[t]=p,e[t+1]=h,e[t+2]=_,e[t+3]=g}static multiplyQuaternionsFlat(e,t,n,a,o,l){const f=n[a],p=n[a+1],h=n[a+2],_=n[a+3],g=o[l],v=o[l+1],b=o[l+2],R=o[l+3];return e[t]=f*R+_*g+p*b-h*v,e[t+1]=p*R+_*v+h*g-f*b,e[t+2]=h*R+_*b+f*v-p*g,e[t+3]=_*R-f*g-p*v-h*b,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,a){return this._x=e,this._y=t,this._z=n,this._w=a,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,a=e._y,o=e._z,l=e._order,f=Math.cos,p=Math.sin,h=f(n/2),_=f(a/2),g=f(o/2),v=p(n/2),b=p(a/2),R=p(o/2);switch(l){case"XYZ":this._x=v*_*g+h*b*R,this._y=h*b*g-v*_*R,this._z=h*_*R+v*b*g,this._w=h*_*g-v*b*R;break;case"YXZ":this._x=v*_*g+h*b*R,this._y=h*b*g-v*_*R,this._z=h*_*R-v*b*g,this._w=h*_*g+v*b*R;break;case"ZXY":this._x=v*_*g-h*b*R,this._y=h*b*g+v*_*R,this._z=h*_*R+v*b*g,this._w=h*_*g-v*b*R;break;case"ZYX":this._x=v*_*g-h*b*R,this._y=h*b*g+v*_*R,this._z=h*_*R-v*b*g,this._w=h*_*g+v*b*R;break;case"YZX":this._x=v*_*g+h*b*R,this._y=h*b*g+v*_*R,this._z=h*_*R-v*b*g,this._w=h*_*g-v*b*R;break;case"XZY":this._x=v*_*g-h*b*R,this._y=h*b*g-v*_*R,this._z=h*_*R+v*b*g,this._w=h*_*g+v*b*R;break;default:Ze("Quaternion: .setFromEuler() encountered an unknown order: "+l)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,a=Math.sin(n);return this._x=e.x*a,this._y=e.y*a,this._z=e.z*a,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],a=t[4],o=t[8],l=t[1],f=t[5],p=t[9],h=t[2],_=t[6],g=t[10],v=n+f+g;if(v>0){const b=.5/Math.sqrt(v+1);this._w=.25/b,this._x=(_-p)*b,this._y=(o-h)*b,this._z=(l-a)*b}else if(n>f&&n>g){const b=2*Math.sqrt(1+n-f-g);this._w=(_-p)/b,this._x=.25*b,this._y=(a+l)/b,this._z=(o+h)/b}else if(f>g){const b=2*Math.sqrt(1+f-n-g);this._w=(o-h)/b,this._x=(a+l)/b,this._y=.25*b,this._z=(p+_)/b}else{const b=2*Math.sqrt(1+g-n-f);this._w=(l-a)/b,this._x=(o+h)/b,this._y=(p+_)/b,this._z=.25*b}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(dt(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const a=Math.min(1,t/n);return this.slerp(e,a),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,a=e._y,o=e._z,l=e._w,f=t._x,p=t._y,h=t._z,_=t._w;return this._x=n*_+l*f+a*h-o*p,this._y=a*_+l*p+o*f-n*h,this._z=o*_+l*h+n*p-a*f,this._w=l*_-n*f-a*p-o*h,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,a=e._y,o=e._z,l=e._w,f=this.dot(e);f<0&&(n=-n,a=-a,o=-o,l=-l,f=-f);let p=1-t;if(f<.9995){const h=Math.acos(f),_=Math.sin(h);p=Math.sin(p*h)/_,t=Math.sin(t*h)/_,this._x=this._x*p+n*t,this._y=this._y*p+a*t,this._z=this._z*p+o*t,this._w=this._w*p+l*t,this._onChangeCallback()}else this._x=this._x*p+n*t,this._y=this._y*p+a*t,this._z=this._z*p+o*t,this._w=this._w*p+l*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),a=Math.sqrt(1-n),o=Math.sqrt(n);return this.set(a*Math.sin(e),a*Math.cos(e),o*Math.sin(t),o*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class K{constructor(e=0,t=0,n=0){K.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(bl.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(bl.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,a=this.z,o=e.elements;return this.x=o[0]*t+o[3]*n+o[6]*a,this.y=o[1]*t+o[4]*n+o[7]*a,this.z=o[2]*t+o[5]*n+o[8]*a,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,a=this.z,o=e.elements,l=1/(o[3]*t+o[7]*n+o[11]*a+o[15]);return this.x=(o[0]*t+o[4]*n+o[8]*a+o[12])*l,this.y=(o[1]*t+o[5]*n+o[9]*a+o[13])*l,this.z=(o[2]*t+o[6]*n+o[10]*a+o[14])*l,this}applyQuaternion(e){const t=this.x,n=this.y,a=this.z,o=e.x,l=e.y,f=e.z,p=e.w,h=2*(l*a-f*n),_=2*(f*t-o*a),g=2*(o*n-l*t);return this.x=t+p*h+l*g-f*_,this.y=n+p*_+f*h-o*g,this.z=a+p*g+o*_-l*h,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,a=this.z,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*a,this.y=o[1]*t+o[5]*n+o[9]*a,this.z=o[2]*t+o[6]*n+o[10]*a,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=dt(this.x,e.x,t.x),this.y=dt(this.y,e.y,t.y),this.z=dt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=dt(this.x,e,t),this.y=dt(this.y,e,t),this.z=dt(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(dt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,a=e.y,o=e.z,l=t.x,f=t.y,p=t.z;return this.x=a*p-o*f,this.y=o*l-n*p,this.z=n*f-a*l,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Na.copy(this).projectOnVector(e),this.sub(Na)}reflect(e){return this.sub(Na.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(dt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,a=this.z-e.z;return t*t+n*n+a*a}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const a=Math.sin(t)*e;return this.x=a*Math.sin(n),this.y=Math.cos(t)*e,this.z=a*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),a=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=a,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Na=new K,bl=new Xi;class tt{constructor(e,t,n,a,o,l,f,p,h){tt.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,a,o,l,f,p,h)}set(e,t,n,a,o,l,f,p,h){const _=this.elements;return _[0]=e,_[1]=a,_[2]=f,_[3]=t,_[4]=o,_[5]=p,_[6]=n,_[7]=l,_[8]=h,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,a=t.elements,o=this.elements,l=n[0],f=n[3],p=n[6],h=n[1],_=n[4],g=n[7],v=n[2],b=n[5],R=n[8],P=a[0],E=a[3],m=a[6],U=a[1],L=a[4],N=a[7],z=a[2],A=a[5],V=a[8];return o[0]=l*P+f*U+p*z,o[3]=l*E+f*L+p*A,o[6]=l*m+f*N+p*V,o[1]=h*P+_*U+g*z,o[4]=h*E+_*L+g*A,o[7]=h*m+_*N+g*V,o[2]=v*P+b*U+R*z,o[5]=v*E+b*L+R*A,o[8]=v*m+b*N+R*V,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],a=e[2],o=e[3],l=e[4],f=e[5],p=e[6],h=e[7],_=e[8];return t*l*_-t*f*h-n*o*_+n*f*p+a*o*h-a*l*p}invert(){const e=this.elements,t=e[0],n=e[1],a=e[2],o=e[3],l=e[4],f=e[5],p=e[6],h=e[7],_=e[8],g=_*l-f*h,v=f*p-_*o,b=h*o-l*p,R=t*g+n*v+a*b;if(R===0)return this.set(0,0,0,0,0,0,0,0,0);const P=1/R;return e[0]=g*P,e[1]=(a*h-_*n)*P,e[2]=(f*n-a*l)*P,e[3]=v*P,e[4]=(_*t-a*p)*P,e[5]=(a*o-f*t)*P,e[6]=b*P,e[7]=(n*p-h*t)*P,e[8]=(l*t-n*o)*P,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,a,o,l,f){const p=Math.cos(o),h=Math.sin(o);return this.set(n*p,n*h,-n*(p*l+h*f)+l+e,-a*h,a*p,-a*(-h*l+p*f)+f+t,0,0,1),this}scale(e,t){return this.premultiply(Oa.makeScale(e,t)),this}rotate(e){return this.premultiply(Oa.makeRotation(-e)),this}translate(e,t){return this.premultiply(Oa.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let a=0;a<9;a++)if(t[a]!==n[a])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const Oa=new tt,El=new tt().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Ml=new tt().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function J0(){const r={enabled:!0,workingColorSpace:Tr,spaces:{},convert:function(a,o,l){return this.enabled===!1||o===l||!o||!l||(this.spaces[o].transfer===Ft&&(a.r=oi(a.r),a.g=oi(a.g),a.b=oi(a.b)),this.spaces[o].primaries!==this.spaces[l].primaries&&(a.applyMatrix3(this.spaces[o].toXYZ),a.applyMatrix3(this.spaces[l].fromXYZ)),this.spaces[l].transfer===Ft&&(a.r=vr(a.r),a.g=vr(a.g),a.b=vr(a.b))),a},workingToColorSpace:function(a,o){return this.convert(a,this.workingColorSpace,o)},colorSpaceToWorking:function(a,o){return this.convert(a,o,this.workingColorSpace)},getPrimaries:function(a){return this.spaces[a].primaries},getTransfer:function(a){return a===yi?ia:this.spaces[a].transfer},getToneMappingMode:function(a){return this.spaces[a].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(a,o=this.workingColorSpace){return a.fromArray(this.spaces[o].luminanceCoefficients)},define:function(a){Object.assign(this.spaces,a)},_getMatrix:function(a,o,l){return a.copy(this.spaces[o].toXYZ).multiply(this.spaces[l].fromXYZ)},_getDrawingBufferColorSpace:function(a){return this.spaces[a].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(a=this.workingColorSpace){return this.spaces[a].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(a,o){return Zr("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),r.workingToColorSpace(a,o)},toWorkingColorSpace:function(a,o){return Zr("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),r.colorSpaceToWorking(a,o)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return r.define({[Tr]:{primaries:e,whitePoint:n,transfer:ia,toXYZ:El,fromXYZ:Ml,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Mn},outputColorSpaceConfig:{drawingBufferColorSpace:Mn}},[Mn]:{primaries:e,whitePoint:n,transfer:Ft,toXYZ:El,fromXYZ:Ml,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Mn}}}),r}const yt=J0();function oi(r){return r<.04045?r*.0773993808:Math.pow(r*.9478672986+.0521327014,2.4)}function vr(r){return r<.0031308?r*12.92:1.055*Math.pow(r,.41666)-.055}let Ji;class Q0{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{Ji===void 0&&(Ji=sa("canvas")),Ji.width=e.width,Ji.height=e.height;const a=Ji.getContext("2d");e instanceof ImageData?a.putImageData(e,0,0):a.drawImage(e,0,0,e.width,e.height),n=Ji}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=sa("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const a=n.getImageData(0,0,e.width,e.height),o=a.data;for(let l=0;l<o.length;l++)o[l]=oi(o[l]/255)*255;return n.putImageData(a,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(oi(t[n]/255)*255):t[n]=oi(t[n]);return{data:t,width:e.width,height:e.height}}else return Ze("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let e_=0;class hc{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:e_++}),this.uuid=ts(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},a=this.data;if(a!==null){let o;if(Array.isArray(a)){o=[];for(let l=0,f=a.length;l<f;l++)a[l].isDataTexture?o.push(ka(a[l].image)):o.push(ka(a[l]))}else o=ka(a);n.url=o}return t||(e.images[this.uuid]=n),n}}function ka(r){return typeof HTMLImageElement<"u"&&r instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&r instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&r instanceof ImageBitmap?Q0.getDataURL(r):r.data?{data:Array.from(r.data),width:r.width,height:r.height,type:r.data.constructor.name}:(Ze("Texture: Unable to serialize Texture."),{})}let t_=0;const Ba=new K;class un extends ji{constructor(e=un.DEFAULT_IMAGE,t=un.DEFAULT_MAPPING,n=ni,a=ni,o=wn,l=Hi,f=Nn,p=$n,h=un.DEFAULT_ANISOTROPY,_=yi){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:t_++}),this.uuid=ts(),this.name="",this.source=new hc(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=a,this.magFilter=o,this.minFilter=l,this.anisotropy=h,this.format=f,this.internalFormat=null,this.type=p,this.offset=new et(0,0),this.repeat=new et(1,1),this.center=new et(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new tt,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=_,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Ba).x}get height(){return this.source.getSize(Ba).y}get depth(){return this.source.getSize(Ba).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){Ze(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const a=this[t];if(a===void 0){Ze(`Texture.setValues(): property '${t}' does not exist.`);continue}a&&n&&a.isVector2&&n.isVector2||a&&n&&a.isVector3&&n.isVector3||a&&n&&a.isMatrix3&&n.isMatrix3?a.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==Eu)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case To:e.x=e.x-Math.floor(e.x);break;case ni:e.x=e.x<0?0:1;break;case wo:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case To:e.y=e.y-Math.floor(e.y);break;case ni:e.y=e.y<0?0:1;break;case wo:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}un.DEFAULT_IMAGE=null;un.DEFAULT_MAPPING=Eu;un.DEFAULT_ANISOTROPY=1;class zt{constructor(e=0,t=0,n=0,a=1){zt.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=a}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,a){return this.x=e,this.y=t,this.z=n,this.w=a,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,a=this.z,o=this.w,l=e.elements;return this.x=l[0]*t+l[4]*n+l[8]*a+l[12]*o,this.y=l[1]*t+l[5]*n+l[9]*a+l[13]*o,this.z=l[2]*t+l[6]*n+l[10]*a+l[14]*o,this.w=l[3]*t+l[7]*n+l[11]*a+l[15]*o,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,a,o;const p=e.elements,h=p[0],_=p[4],g=p[8],v=p[1],b=p[5],R=p[9],P=p[2],E=p[6],m=p[10];if(Math.abs(_-v)<.01&&Math.abs(g-P)<.01&&Math.abs(R-E)<.01){if(Math.abs(_+v)<.1&&Math.abs(g+P)<.1&&Math.abs(R+E)<.1&&Math.abs(h+b+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const L=(h+1)/2,N=(b+1)/2,z=(m+1)/2,A=(_+v)/4,V=(g+P)/4,Z=(R+E)/4;return L>N&&L>z?L<.01?(n=0,a=.707106781,o=.707106781):(n=Math.sqrt(L),a=A/n,o=V/n):N>z?N<.01?(n=.707106781,a=0,o=.707106781):(a=Math.sqrt(N),n=A/a,o=Z/a):z<.01?(n=.707106781,a=.707106781,o=0):(o=Math.sqrt(z),n=V/o,a=Z/o),this.set(n,a,o,t),this}let U=Math.sqrt((E-R)*(E-R)+(g-P)*(g-P)+(v-_)*(v-_));return Math.abs(U)<.001&&(U=1),this.x=(E-R)/U,this.y=(g-P)/U,this.z=(v-_)/U,this.w=Math.acos((h+b+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=dt(this.x,e.x,t.x),this.y=dt(this.y,e.y,t.y),this.z=dt(this.z,e.z,t.z),this.w=dt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=dt(this.x,e,t),this.y=dt(this.y,e,t),this.z=dt(this.z,e,t),this.w=dt(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(dt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class n_ extends ji{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:wn,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new zt(0,0,e,t),this.scissorTest=!1,this.viewport=new zt(0,0,e,t);const a={width:e,height:t,depth:n.depth},o=new un(a);this.textures=[];const l=n.count;for(let f=0;f<l;f++)this.textures[f]=o.clone(),this.textures[f].isRenderTargetTexture=!0,this.textures[f].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:wn,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let a=0,o=this.textures.length;a<o;a++)this.textures[a].image.width=e,this.textures[a].image.height=t,this.textures[a].image.depth=n,this.textures[a].isData3DTexture!==!0&&(this.textures[a].isArrayTexture=this.textures[a].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const a=Object.assign({},e.textures[t].image);this.textures[t].source=new hc(a)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class $i extends n_{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class Iu extends un{constructor(e=null,t=1,n=1,a=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:a},this.magFilter=vn,this.minFilter=vn,this.wrapR=ni,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class i_ extends un{constructor(e=null,t=1,n=1,a=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:a},this.magFilter=vn,this.minFilter=vn,this.wrapR=ni,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class ns{constructor(e=new K(1/0,1/0,1/0),t=new K(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(Cn.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(Cn.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=Cn.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const o=n.getAttribute("position");if(t===!0&&o!==void 0&&e.isInstancedMesh!==!0)for(let l=0,f=o.count;l<f;l++)e.isMesh===!0?e.getVertexPosition(l,Cn):Cn.fromBufferAttribute(o,l),Cn.applyMatrix4(e.matrixWorld),this.expandByPoint(Cn);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),ws.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),ws.copy(n.boundingBox)),ws.applyMatrix4(e.matrixWorld),this.union(ws)}const a=e.children;for(let o=0,l=a.length;o<l;o++)this.expandByObject(a[o],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Cn),Cn.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Br),As.subVectors(this.max,Br),Qi.subVectors(e.a,Br),er.subVectors(e.b,Br),tr.subVectors(e.c,Br),di.subVectors(er,Qi),hi.subVectors(tr,er),Ii.subVectors(Qi,tr);let t=[0,-di.z,di.y,0,-hi.z,hi.y,0,-Ii.z,Ii.y,di.z,0,-di.x,hi.z,0,-hi.x,Ii.z,0,-Ii.x,-di.y,di.x,0,-hi.y,hi.x,0,-Ii.y,Ii.x,0];return!za(t,Qi,er,tr,As)||(t=[1,0,0,0,1,0,0,0,1],!za(t,Qi,er,tr,As))?!1:(Rs.crossVectors(di,hi),t=[Rs.x,Rs.y,Rs.z],za(t,Qi,er,tr,As))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Cn).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Cn).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(qn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),qn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),qn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),qn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),qn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),qn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),qn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),qn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(qn),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const qn=[new K,new K,new K,new K,new K,new K,new K,new K],Cn=new K,ws=new ns,Qi=new K,er=new K,tr=new K,di=new K,hi=new K,Ii=new K,Br=new K,As=new K,Rs=new K,Ui=new K;function za(r,e,t,n,a){for(let o=0,l=r.length-3;o<=l;o+=3){Ui.fromArray(r,o);const f=a.x*Math.abs(Ui.x)+a.y*Math.abs(Ui.y)+a.z*Math.abs(Ui.z),p=e.dot(Ui),h=t.dot(Ui),_=n.dot(Ui);if(Math.max(-Math.max(p,h,_),Math.min(p,h,_))>f)return!1}return!0}const r_=new ns,zr=new K,Va=new K;class fa{constructor(e=new K,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):r_.setFromPoints(e).getCenter(n);let a=0;for(let o=0,l=e.length;o<l;o++)a=Math.max(a,n.distanceToSquared(e[o]));return this.radius=Math.sqrt(a),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;zr.subVectors(e,this.center);const t=zr.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),a=(n-this.radius)*.5;this.center.addScaledVector(zr,a/n),this.radius+=a}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Va.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(zr.copy(e.center).add(Va)),this.expandByPoint(zr.copy(e.center).sub(Va))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const Kn=new K,Ga=new K,Cs=new K,pi=new K,Ha=new K,Ps=new K,Wa=new K;class pc{constructor(e=new K,t=new K(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Kn)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=Kn.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(Kn.copy(this.origin).addScaledVector(this.direction,t),Kn.distanceToSquared(e))}distanceSqToSegment(e,t,n,a){Ga.copy(e).add(t).multiplyScalar(.5),Cs.copy(t).sub(e).normalize(),pi.copy(this.origin).sub(Ga);const o=e.distanceTo(t)*.5,l=-this.direction.dot(Cs),f=pi.dot(this.direction),p=-pi.dot(Cs),h=pi.lengthSq(),_=Math.abs(1-l*l);let g,v,b,R;if(_>0)if(g=l*p-f,v=l*f-p,R=o*_,g>=0)if(v>=-R)if(v<=R){const P=1/_;g*=P,v*=P,b=g*(g+l*v+2*f)+v*(l*g+v+2*p)+h}else v=o,g=Math.max(0,-(l*v+f)),b=-g*g+v*(v+2*p)+h;else v=-o,g=Math.max(0,-(l*v+f)),b=-g*g+v*(v+2*p)+h;else v<=-R?(g=Math.max(0,-(-l*o+f)),v=g>0?-o:Math.min(Math.max(-o,-p),o),b=-g*g+v*(v+2*p)+h):v<=R?(g=0,v=Math.min(Math.max(-o,-p),o),b=v*(v+2*p)+h):(g=Math.max(0,-(l*o+f)),v=g>0?o:Math.min(Math.max(-o,-p),o),b=-g*g+v*(v+2*p)+h);else v=l>0?-o:o,g=Math.max(0,-(l*v+f)),b=-g*g+v*(v+2*p)+h;return n&&n.copy(this.origin).addScaledVector(this.direction,g),a&&a.copy(Ga).addScaledVector(Cs,v),b}intersectSphere(e,t){Kn.subVectors(e.center,this.origin);const n=Kn.dot(this.direction),a=Kn.dot(Kn)-n*n,o=e.radius*e.radius;if(a>o)return null;const l=Math.sqrt(o-a),f=n-l,p=n+l;return p<0?null:f<0?this.at(p,t):this.at(f,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,a,o,l,f,p;const h=1/this.direction.x,_=1/this.direction.y,g=1/this.direction.z,v=this.origin;return h>=0?(n=(e.min.x-v.x)*h,a=(e.max.x-v.x)*h):(n=(e.max.x-v.x)*h,a=(e.min.x-v.x)*h),_>=0?(o=(e.min.y-v.y)*_,l=(e.max.y-v.y)*_):(o=(e.max.y-v.y)*_,l=(e.min.y-v.y)*_),n>l||o>a||((o>n||isNaN(n))&&(n=o),(l<a||isNaN(a))&&(a=l),g>=0?(f=(e.min.z-v.z)*g,p=(e.max.z-v.z)*g):(f=(e.max.z-v.z)*g,p=(e.min.z-v.z)*g),n>p||f>a)||((f>n||n!==n)&&(n=f),(p<a||a!==a)&&(a=p),a<0)?null:this.at(n>=0?n:a,t)}intersectsBox(e){return this.intersectBox(e,Kn)!==null}intersectTriangle(e,t,n,a,o){Ha.subVectors(t,e),Ps.subVectors(n,e),Wa.crossVectors(Ha,Ps);let l=this.direction.dot(Wa),f;if(l>0){if(a)return null;f=1}else if(l<0)f=-1,l=-l;else return null;pi.subVectors(this.origin,e);const p=f*this.direction.dot(Ps.crossVectors(pi,Ps));if(p<0)return null;const h=f*this.direction.dot(Ha.cross(pi));if(h<0||p+h>l)return null;const _=-f*pi.dot(Wa);return _<0?null:this.at(_/l,o)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class Bt{constructor(e,t,n,a,o,l,f,p,h,_,g,v,b,R,P,E){Bt.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,a,o,l,f,p,h,_,g,v,b,R,P,E)}set(e,t,n,a,o,l,f,p,h,_,g,v,b,R,P,E){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=a,m[1]=o,m[5]=l,m[9]=f,m[13]=p,m[2]=h,m[6]=_,m[10]=g,m[14]=v,m[3]=b,m[7]=R,m[11]=P,m[15]=E,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Bt().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,a=1/nr.setFromMatrixColumn(e,0).length(),o=1/nr.setFromMatrixColumn(e,1).length(),l=1/nr.setFromMatrixColumn(e,2).length();return t[0]=n[0]*a,t[1]=n[1]*a,t[2]=n[2]*a,t[3]=0,t[4]=n[4]*o,t[5]=n[5]*o,t[6]=n[6]*o,t[7]=0,t[8]=n[8]*l,t[9]=n[9]*l,t[10]=n[10]*l,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,a=e.y,o=e.z,l=Math.cos(n),f=Math.sin(n),p=Math.cos(a),h=Math.sin(a),_=Math.cos(o),g=Math.sin(o);if(e.order==="XYZ"){const v=l*_,b=l*g,R=f*_,P=f*g;t[0]=p*_,t[4]=-p*g,t[8]=h,t[1]=b+R*h,t[5]=v-P*h,t[9]=-f*p,t[2]=P-v*h,t[6]=R+b*h,t[10]=l*p}else if(e.order==="YXZ"){const v=p*_,b=p*g,R=h*_,P=h*g;t[0]=v+P*f,t[4]=R*f-b,t[8]=l*h,t[1]=l*g,t[5]=l*_,t[9]=-f,t[2]=b*f-R,t[6]=P+v*f,t[10]=l*p}else if(e.order==="ZXY"){const v=p*_,b=p*g,R=h*_,P=h*g;t[0]=v-P*f,t[4]=-l*g,t[8]=R+b*f,t[1]=b+R*f,t[5]=l*_,t[9]=P-v*f,t[2]=-l*h,t[6]=f,t[10]=l*p}else if(e.order==="ZYX"){const v=l*_,b=l*g,R=f*_,P=f*g;t[0]=p*_,t[4]=R*h-b,t[8]=v*h+P,t[1]=p*g,t[5]=P*h+v,t[9]=b*h-R,t[2]=-h,t[6]=f*p,t[10]=l*p}else if(e.order==="YZX"){const v=l*p,b=l*h,R=f*p,P=f*h;t[0]=p*_,t[4]=P-v*g,t[8]=R*g+b,t[1]=g,t[5]=l*_,t[9]=-f*_,t[2]=-h*_,t[6]=b*g+R,t[10]=v-P*g}else if(e.order==="XZY"){const v=l*p,b=l*h,R=f*p,P=f*h;t[0]=p*_,t[4]=-g,t[8]=h*_,t[1]=v*g+P,t[5]=l*_,t[9]=b*g-R,t[2]=R*g-b,t[6]=f*_,t[10]=P*g+v}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(s_,e,a_)}lookAt(e,t,n){const a=this.elements;return xn.subVectors(e,t),xn.lengthSq()===0&&(xn.z=1),xn.normalize(),mi.crossVectors(n,xn),mi.lengthSq()===0&&(Math.abs(n.z)===1?xn.x+=1e-4:xn.z+=1e-4,xn.normalize(),mi.crossVectors(n,xn)),mi.normalize(),Ds.crossVectors(xn,mi),a[0]=mi.x,a[4]=Ds.x,a[8]=xn.x,a[1]=mi.y,a[5]=Ds.y,a[9]=xn.y,a[2]=mi.z,a[6]=Ds.z,a[10]=xn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,a=t.elements,o=this.elements,l=n[0],f=n[4],p=n[8],h=n[12],_=n[1],g=n[5],v=n[9],b=n[13],R=n[2],P=n[6],E=n[10],m=n[14],U=n[3],L=n[7],N=n[11],z=n[15],A=a[0],V=a[4],Z=a[8],D=a[12],y=a[1],G=a[5],Y=a[9],J=a[13],ee=a[2],fe=a[6],le=a[10],Ee=a[14],se=a[3],Me=a[7],we=a[11],Ge=a[15];return o[0]=l*A+f*y+p*ee+h*se,o[4]=l*V+f*G+p*fe+h*Me,o[8]=l*Z+f*Y+p*le+h*we,o[12]=l*D+f*J+p*Ee+h*Ge,o[1]=_*A+g*y+v*ee+b*se,o[5]=_*V+g*G+v*fe+b*Me,o[9]=_*Z+g*Y+v*le+b*we,o[13]=_*D+g*J+v*Ee+b*Ge,o[2]=R*A+P*y+E*ee+m*se,o[6]=R*V+P*G+E*fe+m*Me,o[10]=R*Z+P*Y+E*le+m*we,o[14]=R*D+P*J+E*Ee+m*Ge,o[3]=U*A+L*y+N*ee+z*se,o[7]=U*V+L*G+N*fe+z*Me,o[11]=U*Z+L*Y+N*le+z*we,o[15]=U*D+L*J+N*Ee+z*Ge,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],a=e[8],o=e[12],l=e[1],f=e[5],p=e[9],h=e[13],_=e[2],g=e[6],v=e[10],b=e[14],R=e[3],P=e[7],E=e[11],m=e[15];return R*(+o*p*g-a*h*g-o*f*v+n*h*v+a*f*b-n*p*b)+P*(+t*p*b-t*h*v+o*l*v-a*l*b+a*h*_-o*p*_)+E*(+t*h*g-t*f*b-o*l*g+n*l*b+o*f*_-n*h*_)+m*(-a*f*_-t*p*g+t*f*v+a*l*g-n*l*v+n*p*_)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const a=this.elements;return e.isVector3?(a[12]=e.x,a[13]=e.y,a[14]=e.z):(a[12]=e,a[13]=t,a[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],a=e[2],o=e[3],l=e[4],f=e[5],p=e[6],h=e[7],_=e[8],g=e[9],v=e[10],b=e[11],R=e[12],P=e[13],E=e[14],m=e[15],U=g*E*h-P*v*h+P*p*b-f*E*b-g*p*m+f*v*m,L=R*v*h-_*E*h-R*p*b+l*E*b+_*p*m-l*v*m,N=_*P*h-R*g*h+R*f*b-l*P*b-_*f*m+l*g*m,z=R*g*p-_*P*p-R*f*v+l*P*v+_*f*E-l*g*E,A=t*U+n*L+a*N+o*z;if(A===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const V=1/A;return e[0]=U*V,e[1]=(P*v*o-g*E*o-P*a*b+n*E*b+g*a*m-n*v*m)*V,e[2]=(f*E*o-P*p*o+P*a*h-n*E*h-f*a*m+n*p*m)*V,e[3]=(g*p*o-f*v*o-g*a*h+n*v*h+f*a*b-n*p*b)*V,e[4]=L*V,e[5]=(_*E*o-R*v*o+R*a*b-t*E*b-_*a*m+t*v*m)*V,e[6]=(R*p*o-l*E*o-R*a*h+t*E*h+l*a*m-t*p*m)*V,e[7]=(l*v*o-_*p*o+_*a*h-t*v*h-l*a*b+t*p*b)*V,e[8]=N*V,e[9]=(R*g*o-_*P*o-R*n*b+t*P*b+_*n*m-t*g*m)*V,e[10]=(l*P*o-R*f*o+R*n*h-t*P*h-l*n*m+t*f*m)*V,e[11]=(_*f*o-l*g*o-_*n*h+t*g*h+l*n*b-t*f*b)*V,e[12]=z*V,e[13]=(_*P*a-R*g*a+R*n*v-t*P*v-_*n*E+t*g*E)*V,e[14]=(R*f*a-l*P*a-R*n*p+t*P*p+l*n*E-t*f*E)*V,e[15]=(l*g*a-_*f*a+_*n*p-t*g*p-l*n*v+t*f*v)*V,this}scale(e){const t=this.elements,n=e.x,a=e.y,o=e.z;return t[0]*=n,t[4]*=a,t[8]*=o,t[1]*=n,t[5]*=a,t[9]*=o,t[2]*=n,t[6]*=a,t[10]*=o,t[3]*=n,t[7]*=a,t[11]*=o,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],a=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,a))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),a=Math.sin(t),o=1-n,l=e.x,f=e.y,p=e.z,h=o*l,_=o*f;return this.set(h*l+n,h*f-a*p,h*p+a*f,0,h*f+a*p,_*f+n,_*p-a*l,0,h*p-a*f,_*p+a*l,o*p*p+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,a,o,l){return this.set(1,n,o,0,e,1,l,0,t,a,1,0,0,0,0,1),this}compose(e,t,n){const a=this.elements,o=t._x,l=t._y,f=t._z,p=t._w,h=o+o,_=l+l,g=f+f,v=o*h,b=o*_,R=o*g,P=l*_,E=l*g,m=f*g,U=p*h,L=p*_,N=p*g,z=n.x,A=n.y,V=n.z;return a[0]=(1-(P+m))*z,a[1]=(b+N)*z,a[2]=(R-L)*z,a[3]=0,a[4]=(b-N)*A,a[5]=(1-(v+m))*A,a[6]=(E+U)*A,a[7]=0,a[8]=(R+L)*V,a[9]=(E-U)*V,a[10]=(1-(v+P))*V,a[11]=0,a[12]=e.x,a[13]=e.y,a[14]=e.z,a[15]=1,this}decompose(e,t,n){const a=this.elements;let o=nr.set(a[0],a[1],a[2]).length();const l=nr.set(a[4],a[5],a[6]).length(),f=nr.set(a[8],a[9],a[10]).length();this.determinant()<0&&(o=-o),e.x=a[12],e.y=a[13],e.z=a[14],Pn.copy(this);const h=1/o,_=1/l,g=1/f;return Pn.elements[0]*=h,Pn.elements[1]*=h,Pn.elements[2]*=h,Pn.elements[4]*=_,Pn.elements[5]*=_,Pn.elements[6]*=_,Pn.elements[8]*=g,Pn.elements[9]*=g,Pn.elements[10]*=g,t.setFromRotationMatrix(Pn),n.x=o,n.y=l,n.z=f,this}makePerspective(e,t,n,a,o,l,f=Wn,p=!1){const h=this.elements,_=2*o/(t-e),g=2*o/(n-a),v=(t+e)/(t-e),b=(n+a)/(n-a);let R,P;if(p)R=o/(l-o),P=l*o/(l-o);else if(f===Wn)R=-(l+o)/(l-o),P=-2*l*o/(l-o);else if(f===ra)R=-l/(l-o),P=-l*o/(l-o);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+f);return h[0]=_,h[4]=0,h[8]=v,h[12]=0,h[1]=0,h[5]=g,h[9]=b,h[13]=0,h[2]=0,h[6]=0,h[10]=R,h[14]=P,h[3]=0,h[7]=0,h[11]=-1,h[15]=0,this}makeOrthographic(e,t,n,a,o,l,f=Wn,p=!1){const h=this.elements,_=2/(t-e),g=2/(n-a),v=-(t+e)/(t-e),b=-(n+a)/(n-a);let R,P;if(p)R=1/(l-o),P=l/(l-o);else if(f===Wn)R=-2/(l-o),P=-(l+o)/(l-o);else if(f===ra)R=-1/(l-o),P=-o/(l-o);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+f);return h[0]=_,h[4]=0,h[8]=0,h[12]=v,h[1]=0,h[5]=g,h[9]=0,h[13]=b,h[2]=0,h[6]=0,h[10]=R,h[14]=P,h[3]=0,h[7]=0,h[11]=0,h[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let a=0;a<16;a++)if(t[a]!==n[a])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const nr=new K,Pn=new Bt,s_=new K(0,0,0),a_=new K(1,1,1),mi=new K,Ds=new K,xn=new K,Tl=new Bt,wl=new Xi;class jn{constructor(e=0,t=0,n=0,a=jn.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=a}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,a=this._order){return this._x=e,this._y=t,this._z=n,this._order=a,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const a=e.elements,o=a[0],l=a[4],f=a[8],p=a[1],h=a[5],_=a[9],g=a[2],v=a[6],b=a[10];switch(t){case"XYZ":this._y=Math.asin(dt(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(-_,b),this._z=Math.atan2(-l,o)):(this._x=Math.atan2(v,h),this._z=0);break;case"YXZ":this._x=Math.asin(-dt(_,-1,1)),Math.abs(_)<.9999999?(this._y=Math.atan2(f,b),this._z=Math.atan2(p,h)):(this._y=Math.atan2(-g,o),this._z=0);break;case"ZXY":this._x=Math.asin(dt(v,-1,1)),Math.abs(v)<.9999999?(this._y=Math.atan2(-g,b),this._z=Math.atan2(-l,h)):(this._y=0,this._z=Math.atan2(p,o));break;case"ZYX":this._y=Math.asin(-dt(g,-1,1)),Math.abs(g)<.9999999?(this._x=Math.atan2(v,b),this._z=Math.atan2(p,o)):(this._x=0,this._z=Math.atan2(-l,h));break;case"YZX":this._z=Math.asin(dt(p,-1,1)),Math.abs(p)<.9999999?(this._x=Math.atan2(-_,h),this._y=Math.atan2(-g,o)):(this._x=0,this._y=Math.atan2(f,b));break;case"XZY":this._z=Math.asin(-dt(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(v,h),this._y=Math.atan2(f,o)):(this._x=Math.atan2(-_,b),this._y=0);break;default:Ze("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return Tl.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Tl,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return wl.setFromEuler(this),this.setFromQuaternion(wl,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}jn.DEFAULT_ORDER="XYZ";class Uu{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let o_=0;const Al=new K,ir=new Xi,Zn=new Bt,Ls=new K,Vr=new K,c_=new K,l_=new Xi,Rl=new K(1,0,0),Cl=new K(0,1,0),Pl=new K(0,0,1),Dl={type:"added"},u_={type:"removed"},rr={type:"childadded",child:null},Xa={type:"childremoved",child:null};class en extends ji{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:o_++}),this.uuid=ts(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=en.DEFAULT_UP.clone();const e=new K,t=new jn,n=new Xi,a=new K(1,1,1);function o(){n.setFromEuler(t,!1)}function l(){t.setFromQuaternion(n,void 0,!1)}t._onChange(o),n._onChange(l),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:a},modelViewMatrix:{value:new Bt},normalMatrix:{value:new tt}}),this.matrix=new Bt,this.matrixWorld=new Bt,this.matrixAutoUpdate=en.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=en.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Uu,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return ir.setFromAxisAngle(e,t),this.quaternion.multiply(ir),this}rotateOnWorldAxis(e,t){return ir.setFromAxisAngle(e,t),this.quaternion.premultiply(ir),this}rotateX(e){return this.rotateOnAxis(Rl,e)}rotateY(e){return this.rotateOnAxis(Cl,e)}rotateZ(e){return this.rotateOnAxis(Pl,e)}translateOnAxis(e,t){return Al.copy(e).applyQuaternion(this.quaternion),this.position.add(Al.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Rl,e)}translateY(e){return this.translateOnAxis(Cl,e)}translateZ(e){return this.translateOnAxis(Pl,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Zn.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Ls.copy(e):Ls.set(e,t,n);const a=this.parent;this.updateWorldMatrix(!0,!1),Vr.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Zn.lookAt(Vr,Ls,this.up):Zn.lookAt(Ls,Vr,this.up),this.quaternion.setFromRotationMatrix(Zn),a&&(Zn.extractRotation(a.matrixWorld),ir.setFromRotationMatrix(Zn),this.quaternion.premultiply(ir.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(Gt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Dl),rr.child=e,this.dispatchEvent(rr),rr.child=null):Gt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(u_),Xa.child=e,this.dispatchEvent(Xa),Xa.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Zn.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Zn.multiply(e.parent.matrixWorld)),e.applyMatrix4(Zn),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Dl),rr.child=e,this.dispatchEvent(rr),rr.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,a=this.children.length;n<a;n++){const l=this.children[n].getObjectByProperty(e,t);if(l!==void 0)return l}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const a=this.children;for(let o=0,l=a.length;o<l;o++)a[o].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Vr,e,c_),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Vr,l_,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,a=t.length;n<a;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,a=t.length;n<a;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,a=t.length;n<a;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const a=this.children;for(let o=0,l=a.length;o<l;o++)a[o].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const a={};a.uuid=this.uuid,a.type=this.type,this.name!==""&&(a.name=this.name),this.castShadow===!0&&(a.castShadow=!0),this.receiveShadow===!0&&(a.receiveShadow=!0),this.visible===!1&&(a.visible=!1),this.frustumCulled===!1&&(a.frustumCulled=!1),this.renderOrder!==0&&(a.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(a.userData=this.userData),a.layers=this.layers.mask,a.matrix=this.matrix.toArray(),a.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(a.matrixAutoUpdate=!1),this.isInstancedMesh&&(a.type="InstancedMesh",a.count=this.count,a.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(a.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(a.type="BatchedMesh",a.perObjectFrustumCulled=this.perObjectFrustumCulled,a.sortObjects=this.sortObjects,a.drawRanges=this._drawRanges,a.reservedRanges=this._reservedRanges,a.geometryInfo=this._geometryInfo.map(f=>({...f,boundingBox:f.boundingBox?f.boundingBox.toJSON():void 0,boundingSphere:f.boundingSphere?f.boundingSphere.toJSON():void 0})),a.instanceInfo=this._instanceInfo.map(f=>({...f})),a.availableInstanceIds=this._availableInstanceIds.slice(),a.availableGeometryIds=this._availableGeometryIds.slice(),a.nextIndexStart=this._nextIndexStart,a.nextVertexStart=this._nextVertexStart,a.geometryCount=this._geometryCount,a.maxInstanceCount=this._maxInstanceCount,a.maxVertexCount=this._maxVertexCount,a.maxIndexCount=this._maxIndexCount,a.geometryInitialized=this._geometryInitialized,a.matricesTexture=this._matricesTexture.toJSON(e),a.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(a.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(a.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(a.boundingBox=this.boundingBox.toJSON()));function o(f,p){return f[p.uuid]===void 0&&(f[p.uuid]=p.toJSON(e)),p.uuid}if(this.isScene)this.background&&(this.background.isColor?a.background=this.background.toJSON():this.background.isTexture&&(a.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(a.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){a.geometry=o(e.geometries,this.geometry);const f=this.geometry.parameters;if(f!==void 0&&f.shapes!==void 0){const p=f.shapes;if(Array.isArray(p))for(let h=0,_=p.length;h<_;h++){const g=p[h];o(e.shapes,g)}else o(e.shapes,p)}}if(this.isSkinnedMesh&&(a.bindMode=this.bindMode,a.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(o(e.skeletons,this.skeleton),a.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const f=[];for(let p=0,h=this.material.length;p<h;p++)f.push(o(e.materials,this.material[p]));a.material=f}else a.material=o(e.materials,this.material);if(this.children.length>0){a.children=[];for(let f=0;f<this.children.length;f++)a.children.push(this.children[f].toJSON(e).object)}if(this.animations.length>0){a.animations=[];for(let f=0;f<this.animations.length;f++){const p=this.animations[f];a.animations.push(o(e.animations,p))}}if(t){const f=l(e.geometries),p=l(e.materials),h=l(e.textures),_=l(e.images),g=l(e.shapes),v=l(e.skeletons),b=l(e.animations),R=l(e.nodes);f.length>0&&(n.geometries=f),p.length>0&&(n.materials=p),h.length>0&&(n.textures=h),_.length>0&&(n.images=_),g.length>0&&(n.shapes=g),v.length>0&&(n.skeletons=v),b.length>0&&(n.animations=b),R.length>0&&(n.nodes=R)}return n.object=a,n;function l(f){const p=[];for(const h in f){const _=f[h];delete _.metadata,p.push(_)}return p}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const a=e.children[n];this.add(a.clone())}return this}}en.DEFAULT_UP=new K(0,1,0);en.DEFAULT_MATRIX_AUTO_UPDATE=!0;en.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Dn=new K,Jn=new K,$a=new K,Qn=new K,sr=new K,ar=new K,Ll=new K,ja=new K,Ya=new K,qa=new K,Ka=new zt,Za=new zt,Ja=new zt;class In{constructor(e=new K,t=new K,n=new K){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,a){a.subVectors(n,t),Dn.subVectors(e,t),a.cross(Dn);const o=a.lengthSq();return o>0?a.multiplyScalar(1/Math.sqrt(o)):a.set(0,0,0)}static getBarycoord(e,t,n,a,o){Dn.subVectors(a,t),Jn.subVectors(n,t),$a.subVectors(e,t);const l=Dn.dot(Dn),f=Dn.dot(Jn),p=Dn.dot($a),h=Jn.dot(Jn),_=Jn.dot($a),g=l*h-f*f;if(g===0)return o.set(0,0,0),null;const v=1/g,b=(h*p-f*_)*v,R=(l*_-f*p)*v;return o.set(1-b-R,R,b)}static containsPoint(e,t,n,a){return this.getBarycoord(e,t,n,a,Qn)===null?!1:Qn.x>=0&&Qn.y>=0&&Qn.x+Qn.y<=1}static getInterpolation(e,t,n,a,o,l,f,p){return this.getBarycoord(e,t,n,a,Qn)===null?(p.x=0,p.y=0,"z"in p&&(p.z=0),"w"in p&&(p.w=0),null):(p.setScalar(0),p.addScaledVector(o,Qn.x),p.addScaledVector(l,Qn.y),p.addScaledVector(f,Qn.z),p)}static getInterpolatedAttribute(e,t,n,a,o,l){return Ka.setScalar(0),Za.setScalar(0),Ja.setScalar(0),Ka.fromBufferAttribute(e,t),Za.fromBufferAttribute(e,n),Ja.fromBufferAttribute(e,a),l.setScalar(0),l.addScaledVector(Ka,o.x),l.addScaledVector(Za,o.y),l.addScaledVector(Ja,o.z),l}static isFrontFacing(e,t,n,a){return Dn.subVectors(n,t),Jn.subVectors(e,t),Dn.cross(Jn).dot(a)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,a){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[a]),this}setFromAttributeAndIndices(e,t,n,a){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,a),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Dn.subVectors(this.c,this.b),Jn.subVectors(this.a,this.b),Dn.cross(Jn).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return In.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return In.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,a,o){return In.getInterpolation(e,this.a,this.b,this.c,t,n,a,o)}containsPoint(e){return In.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return In.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,a=this.b,o=this.c;let l,f;sr.subVectors(a,n),ar.subVectors(o,n),ja.subVectors(e,n);const p=sr.dot(ja),h=ar.dot(ja);if(p<=0&&h<=0)return t.copy(n);Ya.subVectors(e,a);const _=sr.dot(Ya),g=ar.dot(Ya);if(_>=0&&g<=_)return t.copy(a);const v=p*g-_*h;if(v<=0&&p>=0&&_<=0)return l=p/(p-_),t.copy(n).addScaledVector(sr,l);qa.subVectors(e,o);const b=sr.dot(qa),R=ar.dot(qa);if(R>=0&&b<=R)return t.copy(o);const P=b*h-p*R;if(P<=0&&h>=0&&R<=0)return f=h/(h-R),t.copy(n).addScaledVector(ar,f);const E=_*R-b*g;if(E<=0&&g-_>=0&&b-R>=0)return Ll.subVectors(o,a),f=(g-_)/(g-_+(b-R)),t.copy(a).addScaledVector(Ll,f);const m=1/(E+P+v);return l=P*m,f=v*m,t.copy(n).addScaledVector(sr,l).addScaledVector(ar,f)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const Nu={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},_i={h:0,s:0,l:0},Fs={h:0,s:0,l:0};function Qa(r,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?r+(e-r)*6*t:t<1/2?e:t<2/3?r+(e-r)*6*(2/3-t):r}class xt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const a=e;a&&a.isColor?this.copy(a):typeof a=="number"?this.setHex(a):typeof a=="string"&&this.setStyle(a)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Mn){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,yt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,a=yt.workingColorSpace){return this.r=e,this.g=t,this.b=n,yt.colorSpaceToWorking(this,a),this}setHSL(e,t,n,a=yt.workingColorSpace){if(e=K0(e,1),t=dt(t,0,1),n=dt(n,0,1),t===0)this.r=this.g=this.b=n;else{const o=n<=.5?n*(1+t):n+t-n*t,l=2*n-o;this.r=Qa(l,o,e+1/3),this.g=Qa(l,o,e),this.b=Qa(l,o,e-1/3)}return yt.colorSpaceToWorking(this,a),this}setStyle(e,t=Mn){function n(o){o!==void 0&&parseFloat(o)<1&&Ze("Color: Alpha component of "+e+" will be ignored.")}let a;if(a=/^(\w+)\(([^\)]*)\)/.exec(e)){let o;const l=a[1],f=a[2];switch(l){case"rgb":case"rgba":if(o=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(f))return n(o[4]),this.setRGB(Math.min(255,parseInt(o[1],10))/255,Math.min(255,parseInt(o[2],10))/255,Math.min(255,parseInt(o[3],10))/255,t);if(o=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(f))return n(o[4]),this.setRGB(Math.min(100,parseInt(o[1],10))/100,Math.min(100,parseInt(o[2],10))/100,Math.min(100,parseInt(o[3],10))/100,t);break;case"hsl":case"hsla":if(o=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(f))return n(o[4]),this.setHSL(parseFloat(o[1])/360,parseFloat(o[2])/100,parseFloat(o[3])/100,t);break;default:Ze("Color: Unknown color model "+e)}}else if(a=/^\#([A-Fa-f\d]+)$/.exec(e)){const o=a[1],l=o.length;if(l===3)return this.setRGB(parseInt(o.charAt(0),16)/15,parseInt(o.charAt(1),16)/15,parseInt(o.charAt(2),16)/15,t);if(l===6)return this.setHex(parseInt(o,16),t);Ze("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Mn){const n=Nu[e.toLowerCase()];return n!==void 0?this.setHex(n,t):Ze("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=oi(e.r),this.g=oi(e.g),this.b=oi(e.b),this}copyLinearToSRGB(e){return this.r=vr(e.r),this.g=vr(e.g),this.b=vr(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Mn){return yt.workingToColorSpace(an.copy(this),e),Math.round(dt(an.r*255,0,255))*65536+Math.round(dt(an.g*255,0,255))*256+Math.round(dt(an.b*255,0,255))}getHexString(e=Mn){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=yt.workingColorSpace){yt.workingToColorSpace(an.copy(this),t);const n=an.r,a=an.g,o=an.b,l=Math.max(n,a,o),f=Math.min(n,a,o);let p,h;const _=(f+l)/2;if(f===l)p=0,h=0;else{const g=l-f;switch(h=_<=.5?g/(l+f):g/(2-l-f),l){case n:p=(a-o)/g+(a<o?6:0);break;case a:p=(o-n)/g+2;break;case o:p=(n-a)/g+4;break}p/=6}return e.h=p,e.s=h,e.l=_,e}getRGB(e,t=yt.workingColorSpace){return yt.workingToColorSpace(an.copy(this),t),e.r=an.r,e.g=an.g,e.b=an.b,e}getStyle(e=Mn){yt.workingToColorSpace(an.copy(this),e);const t=an.r,n=an.g,a=an.b;return e!==Mn?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${a.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(a*255)})`}offsetHSL(e,t,n){return this.getHSL(_i),this.setHSL(_i.h+e,_i.s+t,_i.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(_i),e.getHSL(Fs);const n=Ua(_i.h,Fs.h,t),a=Ua(_i.s,Fs.s,t),o=Ua(_i.l,Fs.l,t);return this.setHSL(n,a,o),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,a=this.b,o=e.elements;return this.r=o[0]*t+o[3]*n+o[6]*a,this.g=o[1]*t+o[4]*n+o[7]*a,this.b=o[2]*t+o[5]*n+o[8]*a,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const an=new xt;xt.NAMES=Nu;let f_=0;class Cr extends ji{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:f_++}),this.uuid=ts(),this.name="",this.type="Material",this.blending=gr,this.side=Ei,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=po,this.blendDst=mo,this.blendEquation=Vi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new xt(0,0,0),this.blendAlpha=0,this.depthFunc=br,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=xl,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Zi,this.stencilZFail=Zi,this.stencilZPass=Zi,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){Ze(`Material: parameter '${t}' has value of undefined.`);continue}const a=this[t];if(a===void 0){Ze(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}a&&a.isColor?a.set(n):a&&a.isVector3&&n&&n.isVector3?a.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==gr&&(n.blending=this.blending),this.side!==Ei&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==po&&(n.blendSrc=this.blendSrc),this.blendDst!==mo&&(n.blendDst=this.blendDst),this.blendEquation!==Vi&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==br&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==xl&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Zi&&(n.stencilFail=this.stencilFail),this.stencilZFail!==Zi&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==Zi&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function a(o){const l=[];for(const f in o){const p=o[f];delete p.metadata,l.push(p)}return l}if(t){const o=a(e.textures),l=a(e.images);o.length>0&&(n.textures=o),l.length>0&&(n.images=l)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const a=t.length;n=new Array(a);for(let o=0;o!==a;++o)n[o]=t[o].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class Ou extends Cr{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new xt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new jn,this.combine=bu,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Ht=new K,Is=new et;let d_=0;class An{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:d_++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=gl,this.updateRanges=[],this.gpuType=ii,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let a=0,o=this.itemSize;a<o;a++)this.array[e+a]=t.array[n+a];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)Is.fromBufferAttribute(this,t),Is.applyMatrix3(e),this.setXY(t,Is.x,Is.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)Ht.fromBufferAttribute(this,t),Ht.applyMatrix3(e),this.setXYZ(t,Ht.x,Ht.y,Ht.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)Ht.fromBufferAttribute(this,t),Ht.applyMatrix4(e),this.setXYZ(t,Ht.x,Ht.y,Ht.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)Ht.fromBufferAttribute(this,t),Ht.applyNormalMatrix(e),this.setXYZ(t,Ht.x,Ht.y,Ht.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)Ht.fromBufferAttribute(this,t),Ht.transformDirection(e),this.setXYZ(t,Ht.x,Ht.y,Ht.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=kr(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=hn(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=kr(t,this.array)),t}setX(e,t){return this.normalized&&(t=hn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=kr(t,this.array)),t}setY(e,t){return this.normalized&&(t=hn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=kr(t,this.array)),t}setZ(e,t){return this.normalized&&(t=hn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=kr(t,this.array)),t}setW(e,t){return this.normalized&&(t=hn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=hn(t,this.array),n=hn(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,a){return e*=this.itemSize,this.normalized&&(t=hn(t,this.array),n=hn(n,this.array),a=hn(a,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=a,this}setXYZW(e,t,n,a,o){return e*=this.itemSize,this.normalized&&(t=hn(t,this.array),n=hn(n,this.array),a=hn(a,this.array),o=hn(o,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=a,this.array[e+3]=o,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==gl&&(e.usage=this.usage),e}}class ku extends An{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class Bu extends An{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class Zt extends An{constructor(e,t,n){super(new Float32Array(e),t,n)}}let h_=0;const bn=new Bt,eo=new en,or=new K,gn=new ns,Gr=new ns,Kt=new K;class fn extends ji{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:h_++}),this.uuid=ts(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Fu(e)?Bu:ku)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const o=new tt().getNormalMatrix(e);n.applyNormalMatrix(o),n.needsUpdate=!0}const a=this.attributes.tangent;return a!==void 0&&(a.transformDirection(e),a.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return bn.makeRotationFromQuaternion(e),this.applyMatrix4(bn),this}rotateX(e){return bn.makeRotationX(e),this.applyMatrix4(bn),this}rotateY(e){return bn.makeRotationY(e),this.applyMatrix4(bn),this}rotateZ(e){return bn.makeRotationZ(e),this.applyMatrix4(bn),this}translate(e,t,n){return bn.makeTranslation(e,t,n),this.applyMatrix4(bn),this}scale(e,t,n){return bn.makeScale(e,t,n),this.applyMatrix4(bn),this}lookAt(e){return eo.lookAt(e),eo.updateMatrix(),this.applyMatrix4(eo.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(or).negate(),this.translate(or.x,or.y,or.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let a=0,o=e.length;a<o;a++){const l=e[a];n.push(l.x,l.y,l.z||0)}this.setAttribute("position",new Zt(n,3))}else{const n=Math.min(e.length,t.count);for(let a=0;a<n;a++){const o=e[a];t.setXYZ(a,o.x,o.y,o.z||0)}e.length>t.count&&Ze("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new ns);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Gt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new K(-1/0,-1/0,-1/0),new K(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,a=t.length;n<a;n++){const o=t[n];gn.setFromBufferAttribute(o),this.morphTargetsRelative?(Kt.addVectors(this.boundingBox.min,gn.min),this.boundingBox.expandByPoint(Kt),Kt.addVectors(this.boundingBox.max,gn.max),this.boundingBox.expandByPoint(Kt)):(this.boundingBox.expandByPoint(gn.min),this.boundingBox.expandByPoint(gn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Gt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new fa);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Gt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new K,1/0);return}if(e){const n=this.boundingSphere.center;if(gn.setFromBufferAttribute(e),t)for(let o=0,l=t.length;o<l;o++){const f=t[o];Gr.setFromBufferAttribute(f),this.morphTargetsRelative?(Kt.addVectors(gn.min,Gr.min),gn.expandByPoint(Kt),Kt.addVectors(gn.max,Gr.max),gn.expandByPoint(Kt)):(gn.expandByPoint(Gr.min),gn.expandByPoint(Gr.max))}gn.getCenter(n);let a=0;for(let o=0,l=e.count;o<l;o++)Kt.fromBufferAttribute(e,o),a=Math.max(a,n.distanceToSquared(Kt));if(t)for(let o=0,l=t.length;o<l;o++){const f=t[o],p=this.morphTargetsRelative;for(let h=0,_=f.count;h<_;h++)Kt.fromBufferAttribute(f,h),p&&(or.fromBufferAttribute(e,h),Kt.add(or)),a=Math.max(a,n.distanceToSquared(Kt))}this.boundingSphere.radius=Math.sqrt(a),isNaN(this.boundingSphere.radius)&&Gt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){Gt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,a=t.normal,o=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new An(new Float32Array(4*n.count),4));const l=this.getAttribute("tangent"),f=[],p=[];for(let Z=0;Z<n.count;Z++)f[Z]=new K,p[Z]=new K;const h=new K,_=new K,g=new K,v=new et,b=new et,R=new et,P=new K,E=new K;function m(Z,D,y){h.fromBufferAttribute(n,Z),_.fromBufferAttribute(n,D),g.fromBufferAttribute(n,y),v.fromBufferAttribute(o,Z),b.fromBufferAttribute(o,D),R.fromBufferAttribute(o,y),_.sub(h),g.sub(h),b.sub(v),R.sub(v);const G=1/(b.x*R.y-R.x*b.y);isFinite(G)&&(P.copy(_).multiplyScalar(R.y).addScaledVector(g,-b.y).multiplyScalar(G),E.copy(g).multiplyScalar(b.x).addScaledVector(_,-R.x).multiplyScalar(G),f[Z].add(P),f[D].add(P),f[y].add(P),p[Z].add(E),p[D].add(E),p[y].add(E))}let U=this.groups;U.length===0&&(U=[{start:0,count:e.count}]);for(let Z=0,D=U.length;Z<D;++Z){const y=U[Z],G=y.start,Y=y.count;for(let J=G,ee=G+Y;J<ee;J+=3)m(e.getX(J+0),e.getX(J+1),e.getX(J+2))}const L=new K,N=new K,z=new K,A=new K;function V(Z){z.fromBufferAttribute(a,Z),A.copy(z);const D=f[Z];L.copy(D),L.sub(z.multiplyScalar(z.dot(D))).normalize(),N.crossVectors(A,D);const G=N.dot(p[Z])<0?-1:1;l.setXYZW(Z,L.x,L.y,L.z,G)}for(let Z=0,D=U.length;Z<D;++Z){const y=U[Z],G=y.start,Y=y.count;for(let J=G,ee=G+Y;J<ee;J+=3)V(e.getX(J+0)),V(e.getX(J+1)),V(e.getX(J+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new An(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let v=0,b=n.count;v<b;v++)n.setXYZ(v,0,0,0);const a=new K,o=new K,l=new K,f=new K,p=new K,h=new K,_=new K,g=new K;if(e)for(let v=0,b=e.count;v<b;v+=3){const R=e.getX(v+0),P=e.getX(v+1),E=e.getX(v+2);a.fromBufferAttribute(t,R),o.fromBufferAttribute(t,P),l.fromBufferAttribute(t,E),_.subVectors(l,o),g.subVectors(a,o),_.cross(g),f.fromBufferAttribute(n,R),p.fromBufferAttribute(n,P),h.fromBufferAttribute(n,E),f.add(_),p.add(_),h.add(_),n.setXYZ(R,f.x,f.y,f.z),n.setXYZ(P,p.x,p.y,p.z),n.setXYZ(E,h.x,h.y,h.z)}else for(let v=0,b=t.count;v<b;v+=3)a.fromBufferAttribute(t,v+0),o.fromBufferAttribute(t,v+1),l.fromBufferAttribute(t,v+2),_.subVectors(l,o),g.subVectors(a,o),_.cross(g),n.setXYZ(v+0,_.x,_.y,_.z),n.setXYZ(v+1,_.x,_.y,_.z),n.setXYZ(v+2,_.x,_.y,_.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)Kt.fromBufferAttribute(e,t),Kt.normalize(),e.setXYZ(t,Kt.x,Kt.y,Kt.z)}toNonIndexed(){function e(f,p){const h=f.array,_=f.itemSize,g=f.normalized,v=new h.constructor(p.length*_);let b=0,R=0;for(let P=0,E=p.length;P<E;P++){f.isInterleavedBufferAttribute?b=p[P]*f.data.stride+f.offset:b=p[P]*_;for(let m=0;m<_;m++)v[R++]=h[b++]}return new An(v,_,g)}if(this.index===null)return Ze("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new fn,n=this.index.array,a=this.attributes;for(const f in a){const p=a[f],h=e(p,n);t.setAttribute(f,h)}const o=this.morphAttributes;for(const f in o){const p=[],h=o[f];for(let _=0,g=h.length;_<g;_++){const v=h[_],b=e(v,n);p.push(b)}t.morphAttributes[f]=p}t.morphTargetsRelative=this.morphTargetsRelative;const l=this.groups;for(let f=0,p=l.length;f<p;f++){const h=l[f];t.addGroup(h.start,h.count,h.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const p=this.parameters;for(const h in p)p[h]!==void 0&&(e[h]=p[h]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const p in n){const h=n[p];e.data.attributes[p]=h.toJSON(e.data)}const a={};let o=!1;for(const p in this.morphAttributes){const h=this.morphAttributes[p],_=[];for(let g=0,v=h.length;g<v;g++){const b=h[g];_.push(b.toJSON(e.data))}_.length>0&&(a[p]=_,o=!0)}o&&(e.data.morphAttributes=a,e.data.morphTargetsRelative=this.morphTargetsRelative);const l=this.groups;l.length>0&&(e.data.groups=JSON.parse(JSON.stringify(l)));const f=this.boundingSphere;return f!==null&&(e.data.boundingSphere=f.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const a=e.attributes;for(const h in a){const _=a[h];this.setAttribute(h,_.clone(t))}const o=e.morphAttributes;for(const h in o){const _=[],g=o[h];for(let v=0,b=g.length;v<b;v++)_.push(g[v].clone(t));this.morphAttributes[h]=_}this.morphTargetsRelative=e.morphTargetsRelative;const l=e.groups;for(let h=0,_=l.length;h<_;h++){const g=l[h];this.addGroup(g.start,g.count,g.materialIndex)}const f=e.boundingBox;f!==null&&(this.boundingBox=f.clone());const p=e.boundingSphere;return p!==null&&(this.boundingSphere=p.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const Fl=new Bt,Ni=new pc,Us=new fa,Il=new K,Ns=new K,Os=new K,ks=new K,to=new K,Bs=new K,Ul=new K,zs=new K;class li extends en{constructor(e=new fn,t=new Ou){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const a=t[n[0]];if(a!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let o=0,l=a.length;o<l;o++){const f=a[o].name||String(o);this.morphTargetInfluences.push(0),this.morphTargetDictionary[f]=o}}}}getVertexPosition(e,t){const n=this.geometry,a=n.attributes.position,o=n.morphAttributes.position,l=n.morphTargetsRelative;t.fromBufferAttribute(a,e);const f=this.morphTargetInfluences;if(o&&f){Bs.set(0,0,0);for(let p=0,h=o.length;p<h;p++){const _=f[p],g=o[p];_!==0&&(to.fromBufferAttribute(g,e),l?Bs.addScaledVector(to,_):Bs.addScaledVector(to.sub(t),_))}t.add(Bs)}return t}raycast(e,t){const n=this.geometry,a=this.material,o=this.matrixWorld;a!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Us.copy(n.boundingSphere),Us.applyMatrix4(o),Ni.copy(e.ray).recast(e.near),!(Us.containsPoint(Ni.origin)===!1&&(Ni.intersectSphere(Us,Il)===null||Ni.origin.distanceToSquared(Il)>(e.far-e.near)**2))&&(Fl.copy(o).invert(),Ni.copy(e.ray).applyMatrix4(Fl),!(n.boundingBox!==null&&Ni.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,Ni)))}_computeIntersections(e,t,n){let a;const o=this.geometry,l=this.material,f=o.index,p=o.attributes.position,h=o.attributes.uv,_=o.attributes.uv1,g=o.attributes.normal,v=o.groups,b=o.drawRange;if(f!==null)if(Array.isArray(l))for(let R=0,P=v.length;R<P;R++){const E=v[R],m=l[E.materialIndex],U=Math.max(E.start,b.start),L=Math.min(f.count,Math.min(E.start+E.count,b.start+b.count));for(let N=U,z=L;N<z;N+=3){const A=f.getX(N),V=f.getX(N+1),Z=f.getX(N+2);a=Vs(this,m,e,n,h,_,g,A,V,Z),a&&(a.faceIndex=Math.floor(N/3),a.face.materialIndex=E.materialIndex,t.push(a))}}else{const R=Math.max(0,b.start),P=Math.min(f.count,b.start+b.count);for(let E=R,m=P;E<m;E+=3){const U=f.getX(E),L=f.getX(E+1),N=f.getX(E+2);a=Vs(this,l,e,n,h,_,g,U,L,N),a&&(a.faceIndex=Math.floor(E/3),t.push(a))}}else if(p!==void 0)if(Array.isArray(l))for(let R=0,P=v.length;R<P;R++){const E=v[R],m=l[E.materialIndex],U=Math.max(E.start,b.start),L=Math.min(p.count,Math.min(E.start+E.count,b.start+b.count));for(let N=U,z=L;N<z;N+=3){const A=N,V=N+1,Z=N+2;a=Vs(this,m,e,n,h,_,g,A,V,Z),a&&(a.faceIndex=Math.floor(N/3),a.face.materialIndex=E.materialIndex,t.push(a))}}else{const R=Math.max(0,b.start),P=Math.min(p.count,b.start+b.count);for(let E=R,m=P;E<m;E+=3){const U=E,L=E+1,N=E+2;a=Vs(this,l,e,n,h,_,g,U,L,N),a&&(a.faceIndex=Math.floor(E/3),t.push(a))}}}}function p_(r,e,t,n,a,o,l,f){let p;if(e.side===_n?p=n.intersectTriangle(l,o,a,!0,f):p=n.intersectTriangle(a,o,l,e.side===Ei,f),p===null)return null;zs.copy(f),zs.applyMatrix4(r.matrixWorld);const h=t.ray.origin.distanceTo(zs);return h<t.near||h>t.far?null:{distance:h,point:zs.clone(),object:r}}function Vs(r,e,t,n,a,o,l,f,p,h){r.getVertexPosition(f,Ns),r.getVertexPosition(p,Os),r.getVertexPosition(h,ks);const _=p_(r,e,t,n,Ns,Os,ks,Ul);if(_){const g=new K;In.getBarycoord(Ul,Ns,Os,ks,g),a&&(_.uv=In.getInterpolatedAttribute(a,f,p,h,g,new et)),o&&(_.uv1=In.getInterpolatedAttribute(o,f,p,h,g,new et)),l&&(_.normal=In.getInterpolatedAttribute(l,f,p,h,g,new K),_.normal.dot(n.direction)>0&&_.normal.multiplyScalar(-1));const v={a:f,b:p,c:h,normal:new K,materialIndex:0};In.getNormal(Ns,Os,ks,v.normal),_.face=v,_.barycoord=g}return _}class Pr extends fn{constructor(e=1,t=1,n=1,a=1,o=1,l=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:a,heightSegments:o,depthSegments:l};const f=this;a=Math.floor(a),o=Math.floor(o),l=Math.floor(l);const p=[],h=[],_=[],g=[];let v=0,b=0;R("z","y","x",-1,-1,n,t,e,l,o,0),R("z","y","x",1,-1,n,t,-e,l,o,1),R("x","z","y",1,1,e,n,t,a,l,2),R("x","z","y",1,-1,e,n,-t,a,l,3),R("x","y","z",1,-1,e,t,n,a,o,4),R("x","y","z",-1,-1,e,t,-n,a,o,5),this.setIndex(p),this.setAttribute("position",new Zt(h,3)),this.setAttribute("normal",new Zt(_,3)),this.setAttribute("uv",new Zt(g,2));function R(P,E,m,U,L,N,z,A,V,Z,D){const y=N/V,G=z/Z,Y=N/2,J=z/2,ee=A/2,fe=V+1,le=Z+1;let Ee=0,se=0;const Me=new K;for(let we=0;we<le;we++){const Ge=we*G-J;for(let ot=0;ot<fe;ot++){const _t=ot*y-Y;Me[P]=_t*U,Me[E]=Ge*L,Me[m]=ee,h.push(Me.x,Me.y,Me.z),Me[P]=0,Me[E]=0,Me[m]=A>0?1:-1,_.push(Me.x,Me.y,Me.z),g.push(ot/V),g.push(1-we/Z),Ee+=1}}for(let we=0;we<Z;we++)for(let Ge=0;Ge<V;Ge++){const ot=v+Ge+fe*we,_t=v+Ge+fe*(we+1),Je=v+(Ge+1)+fe*(we+1),at=v+(Ge+1)+fe*we;p.push(ot,_t,at),p.push(_t,Je,at),se+=6}f.addGroup(b,se,D),b+=se,v+=Ee}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Pr(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function wr(r){const e={};for(const t in r){e[t]={};for(const n in r[t]){const a=r[t][n];a&&(a.isColor||a.isMatrix3||a.isMatrix4||a.isVector2||a.isVector3||a.isVector4||a.isTexture||a.isQuaternion)?a.isRenderTargetTexture?(Ze("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=a.clone():Array.isArray(a)?e[t][n]=a.slice():e[t][n]=a}}return e}function ln(r){const e={};for(let t=0;t<r.length;t++){const n=wr(r[t]);for(const a in n)e[a]=n[a]}return e}function m_(r){const e=[];for(let t=0;t<r.length;t++)e.push(r[t].clone());return e}function zu(r){const e=r.getRenderTarget();return e===null?r.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:yt.workingColorSpace}const __={clone:wr,merge:ln};var x_=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,g_=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class ui extends Cr{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=x_,this.fragmentShader=g_,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=wr(e.uniforms),this.uniformsGroups=m_(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const a in this.uniforms){const l=this.uniforms[a].value;l&&l.isTexture?t.uniforms[a]={type:"t",value:l.toJSON(e).uuid}:l&&l.isColor?t.uniforms[a]={type:"c",value:l.getHex()}:l&&l.isVector2?t.uniforms[a]={type:"v2",value:l.toArray()}:l&&l.isVector3?t.uniforms[a]={type:"v3",value:l.toArray()}:l&&l.isVector4?t.uniforms[a]={type:"v4",value:l.toArray()}:l&&l.isMatrix3?t.uniforms[a]={type:"m3",value:l.toArray()}:l&&l.isMatrix4?t.uniforms[a]={type:"m4",value:l.toArray()}:t.uniforms[a]={value:l}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const a in this.extensions)this.extensions[a]===!0&&(n[a]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class Vu extends en{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Bt,this.projectionMatrix=new Bt,this.projectionMatrixInverse=new Bt,this.coordinateSystem=Wn,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const xi=new K,Nl=new et,Ol=new et;class Tn extends Vu{constructor(e=50,t=1,n=.1,a=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=a,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=tc*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(ea*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return tc*2*Math.atan(Math.tan(ea*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){xi.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(xi.x,xi.y).multiplyScalar(-e/xi.z),xi.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(xi.x,xi.y).multiplyScalar(-e/xi.z)}getViewSize(e,t){return this.getViewBounds(e,Nl,Ol),t.subVectors(Ol,Nl)}setViewOffset(e,t,n,a,o,l){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=a,this.view.width=o,this.view.height=l,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(ea*.5*this.fov)/this.zoom,n=2*t,a=this.aspect*n,o=-.5*a;const l=this.view;if(this.view!==null&&this.view.enabled){const p=l.fullWidth,h=l.fullHeight;o+=l.offsetX*a/p,t-=l.offsetY*n/h,a*=l.width/p,n*=l.height/h}const f=this.filmOffset;f!==0&&(o+=e*f/this.getFilmWidth()),this.projectionMatrix.makePerspective(o,o+a,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const cr=-90,lr=1;class v_ extends en{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const a=new Tn(cr,lr,e,t);a.layers=this.layers,this.add(a);const o=new Tn(cr,lr,e,t);o.layers=this.layers,this.add(o);const l=new Tn(cr,lr,e,t);l.layers=this.layers,this.add(l);const f=new Tn(cr,lr,e,t);f.layers=this.layers,this.add(f);const p=new Tn(cr,lr,e,t);p.layers=this.layers,this.add(p);const h=new Tn(cr,lr,e,t);h.layers=this.layers,this.add(h)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,a,o,l,f,p]=t;for(const h of t)this.remove(h);if(e===Wn)n.up.set(0,1,0),n.lookAt(1,0,0),a.up.set(0,1,0),a.lookAt(-1,0,0),o.up.set(0,0,-1),o.lookAt(0,1,0),l.up.set(0,0,1),l.lookAt(0,-1,0),f.up.set(0,1,0),f.lookAt(0,0,1),p.up.set(0,1,0),p.lookAt(0,0,-1);else if(e===ra)n.up.set(0,-1,0),n.lookAt(-1,0,0),a.up.set(0,-1,0),a.lookAt(1,0,0),o.up.set(0,0,1),o.lookAt(0,1,0),l.up.set(0,0,-1),l.lookAt(0,-1,0),f.up.set(0,-1,0),f.lookAt(0,0,1),p.up.set(0,-1,0),p.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const h of t)this.add(h),h.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:a}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[o,l,f,p,h,_]=this.children,g=e.getRenderTarget(),v=e.getActiveCubeFace(),b=e.getActiveMipmapLevel(),R=e.xr.enabled;e.xr.enabled=!1;const P=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,a),e.render(t,o),e.setRenderTarget(n,1,a),e.render(t,l),e.setRenderTarget(n,2,a),e.render(t,f),e.setRenderTarget(n,3,a),e.render(t,p),e.setRenderTarget(n,4,a),e.render(t,h),n.texture.generateMipmaps=P,e.setRenderTarget(n,5,a),e.render(t,_),e.setRenderTarget(g,v,b),e.xr.enabled=R,n.texture.needsPMREMUpdate=!0}}class Gu extends un{constructor(e=[],t=Er,n,a,o,l,f,p,h,_){super(e,t,n,a,o,l,f,p,h,_),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class y_ extends $i{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},a=[n,n,n,n,n,n];this.texture=new Gu(a),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},a=new Pr(5,5,5),o=new ui({name:"CubemapFromEquirect",uniforms:wr(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:_n,blending:ai});o.uniforms.tEquirect.value=t;const l=new li(a,o),f=t.minFilter;return t.minFilter===Hi&&(t.minFilter=wn),new v_(1,10,this).update(e,l),t.minFilter=f,l.geometry.dispose(),l.material.dispose(),this}clear(e,t=!0,n=!0,a=!0){const o=e.getRenderTarget();for(let l=0;l<6;l++)e.setRenderTarget(this,l),e.clear(t,n,a);e.setRenderTarget(o)}}class Gs extends en{constructor(){super(),this.isGroup=!0,this.type="Group"}}const S_={type:"move"};class no{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Gs,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Gs,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new K,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new K),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Gs,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new K,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new K),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let a=null,o=null,l=null;const f=this._targetRay,p=this._grip,h=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(h&&e.hand){l=!0;for(const P of e.hand.values()){const E=t.getJointPose(P,n),m=this._getHandJoint(h,P);E!==null&&(m.matrix.fromArray(E.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=E.radius),m.visible=E!==null}const _=h.joints["index-finger-tip"],g=h.joints["thumb-tip"],v=_.position.distanceTo(g.position),b=.02,R=.005;h.inputState.pinching&&v>b+R?(h.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!h.inputState.pinching&&v<=b-R&&(h.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else p!==null&&e.gripSpace&&(o=t.getPose(e.gripSpace,n),o!==null&&(p.matrix.fromArray(o.transform.matrix),p.matrix.decompose(p.position,p.rotation,p.scale),p.matrixWorldNeedsUpdate=!0,o.linearVelocity?(p.hasLinearVelocity=!0,p.linearVelocity.copy(o.linearVelocity)):p.hasLinearVelocity=!1,o.angularVelocity?(p.hasAngularVelocity=!0,p.angularVelocity.copy(o.angularVelocity)):p.hasAngularVelocity=!1));f!==null&&(a=t.getPose(e.targetRaySpace,n),a===null&&o!==null&&(a=o),a!==null&&(f.matrix.fromArray(a.transform.matrix),f.matrix.decompose(f.position,f.rotation,f.scale),f.matrixWorldNeedsUpdate=!0,a.linearVelocity?(f.hasLinearVelocity=!0,f.linearVelocity.copy(a.linearVelocity)):f.hasLinearVelocity=!1,a.angularVelocity?(f.hasAngularVelocity=!0,f.angularVelocity.copy(a.angularVelocity)):f.hasAngularVelocity=!1,this.dispatchEvent(S_)))}return f!==null&&(f.visible=a!==null),p!==null&&(p.visible=o!==null),h!==null&&(h.visible=l!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new Gs;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class b_ extends en{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new jn,this.environmentIntensity=1,this.environmentRotation=new jn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class E_ extends un{constructor(e=null,t=1,n=1,a,o,l,f,p,h=vn,_=vn,g,v){super(null,l,f,p,h,_,a,o,g,v),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}const io=new K,M_=new K,T_=new tt;class vi{constructor(e=new K(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,a){return this.normal.set(e,t,n),this.constant=a,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const a=io.subVectors(n,t).cross(M_.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(a,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(io),a=this.normal.dot(n);if(a===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const o=-(e.start.dot(this.normal)+this.constant)/a;return o<0||o>1?null:t.copy(e.start).addScaledVector(n,o)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||T_.getNormalMatrix(e),a=this.coplanarPoint(io).applyMatrix4(e),o=this.normal.applyMatrix3(n).normalize();return this.constant=-a.dot(o),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Oi=new fa,w_=new et(.5,.5),Hs=new K;class mc{constructor(e=new vi,t=new vi,n=new vi,a=new vi,o=new vi,l=new vi){this.planes=[e,t,n,a,o,l]}set(e,t,n,a,o,l){const f=this.planes;return f[0].copy(e),f[1].copy(t),f[2].copy(n),f[3].copy(a),f[4].copy(o),f[5].copy(l),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Wn,n=!1){const a=this.planes,o=e.elements,l=o[0],f=o[1],p=o[2],h=o[3],_=o[4],g=o[5],v=o[6],b=o[7],R=o[8],P=o[9],E=o[10],m=o[11],U=o[12],L=o[13],N=o[14],z=o[15];if(a[0].setComponents(h-l,b-_,m-R,z-U).normalize(),a[1].setComponents(h+l,b+_,m+R,z+U).normalize(),a[2].setComponents(h+f,b+g,m+P,z+L).normalize(),a[3].setComponents(h-f,b-g,m-P,z-L).normalize(),n)a[4].setComponents(p,v,E,N).normalize(),a[5].setComponents(h-p,b-v,m-E,z-N).normalize();else if(a[4].setComponents(h-p,b-v,m-E,z-N).normalize(),t===Wn)a[5].setComponents(h+p,b+v,m+E,z+N).normalize();else if(t===ra)a[5].setComponents(p,v,E,N).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Oi.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Oi.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Oi)}intersectsSprite(e){Oi.center.set(0,0,0);const t=w_.distanceTo(e.center);return Oi.radius=.7071067811865476+t,Oi.applyMatrix4(e.matrixWorld),this.intersectsSphere(Oi)}intersectsSphere(e){const t=this.planes,n=e.center,a=-e.radius;for(let o=0;o<6;o++)if(t[o].distanceToPoint(n)<a)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const a=t[n];if(Hs.x=a.normal.x>0?e.max.x:e.min.x,Hs.y=a.normal.y>0?e.max.y:e.min.y,Hs.z=a.normal.z>0?e.max.z:e.min.z,a.distanceToPoint(Hs)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class _c extends Cr{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new xt(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}}const aa=new K,oa=new K,kl=new Bt,Hr=new pc,Ws=new fa,ro=new K,Bl=new K;class Hu extends en{constructor(e=new fn,t=new _c){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){const e=this.geometry;if(e.index===null){const t=e.attributes.position,n=[0];for(let a=1,o=t.count;a<o;a++)aa.fromBufferAttribute(t,a-1),oa.fromBufferAttribute(t,a),n[a]=n[a-1],n[a]+=aa.distanceTo(oa);e.setAttribute("lineDistance",new Zt(n,1))}else Ze("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,t){const n=this.geometry,a=this.matrixWorld,o=e.params.Line.threshold,l=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Ws.copy(n.boundingSphere),Ws.applyMatrix4(a),Ws.radius+=o,e.ray.intersectsSphere(Ws)===!1)return;kl.copy(a).invert(),Hr.copy(e.ray).applyMatrix4(kl);const f=o/((this.scale.x+this.scale.y+this.scale.z)/3),p=f*f,h=this.isLineSegments?2:1,_=n.index,v=n.attributes.position;if(_!==null){const b=Math.max(0,l.start),R=Math.min(_.count,l.start+l.count);for(let P=b,E=R-1;P<E;P+=h){const m=_.getX(P),U=_.getX(P+1),L=Xs(this,e,Hr,p,m,U,P);L&&t.push(L)}if(this.isLineLoop){const P=_.getX(R-1),E=_.getX(b),m=Xs(this,e,Hr,p,P,E,R-1);m&&t.push(m)}}else{const b=Math.max(0,l.start),R=Math.min(v.count,l.start+l.count);for(let P=b,E=R-1;P<E;P+=h){const m=Xs(this,e,Hr,p,P,P+1,P);m&&t.push(m)}if(this.isLineLoop){const P=Xs(this,e,Hr,p,R-1,b,R-1);P&&t.push(P)}}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const a=t[n[0]];if(a!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let o=0,l=a.length;o<l;o++){const f=a[o].name||String(o);this.morphTargetInfluences.push(0),this.morphTargetDictionary[f]=o}}}}}function Xs(r,e,t,n,a,o,l){const f=r.geometry.attributes.position;if(aa.fromBufferAttribute(f,a),oa.fromBufferAttribute(f,o),t.distanceSqToSegment(aa,oa,ro,Bl)>n)return;ro.applyMatrix4(r.matrixWorld);const h=e.ray.origin.distanceTo(ro);if(!(h<e.near||h>e.far))return{distance:h,point:Bl.clone().applyMatrix4(r.matrixWorld),index:l,face:null,faceIndex:null,barycoord:null,object:r}}const zl=new K,Vl=new K;class A_ extends Hu{constructor(e,t){super(e,t),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){const e=this.geometry;if(e.index===null){const t=e.attributes.position,n=[];for(let a=0,o=t.count;a<o;a+=2)zl.fromBufferAttribute(t,a),Vl.fromBufferAttribute(t,a+1),n[a]=a===0?0:n[a-1],n[a+1]=n[a]+zl.distanceTo(Vl);e.setAttribute("lineDistance",new Zt(n,1))}else Ze("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}}class Wu extends un{constructor(e,t,n=Wi,a,o,l,f=vn,p=vn,h,_=qr,g=1){if(_!==qr&&_!==Kr)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const v={width:e,height:t,depth:g};super(v,a,o,l,f,p,_,n,h),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new hc(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class Xu extends un{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class xc extends fn{constructor(e=1,t=1,n=4,a=8,o=1){super(),this.type="CapsuleGeometry",this.parameters={radius:e,height:t,capSegments:n,radialSegments:a,heightSegments:o},t=Math.max(0,t),n=Math.max(1,Math.floor(n)),a=Math.max(3,Math.floor(a)),o=Math.max(1,Math.floor(o));const l=[],f=[],p=[],h=[],_=t/2,g=Math.PI/2*e,v=t,b=2*g+v,R=n*2+o,P=a+1,E=new K,m=new K;for(let U=0;U<=R;U++){let L=0,N=0,z=0,A=0;if(U<=n){const D=U/n,y=D*Math.PI/2;N=-_-e*Math.cos(y),z=e*Math.sin(y),A=-e*Math.cos(y),L=D*g}else if(U<=n+o){const D=(U-n)/o;N=-_+D*t,z=e,A=0,L=g+D*v}else{const D=(U-n-o)/n,y=D*Math.PI/2;N=_+e*Math.sin(y),z=e*Math.cos(y),A=e*Math.sin(y),L=g+v+D*g}const V=Math.max(0,Math.min(1,L/b));let Z=0;U===0?Z=.5/a:U===R&&(Z=-.5/a);for(let D=0;D<=a;D++){const y=D/a,G=y*Math.PI*2,Y=Math.sin(G),J=Math.cos(G);m.x=-z*J,m.y=N,m.z=z*Y,f.push(m.x,m.y,m.z),E.set(-z*J,A,z*Y),E.normalize(),p.push(E.x,E.y,E.z),h.push(y+Z,V)}if(U>0){const D=(U-1)*P;for(let y=0;y<a;y++){const G=D+y,Y=D+y+1,J=U*P+y,ee=U*P+y+1;l.push(G,Y,J),l.push(Y,ee,J)}}}this.setIndex(l),this.setAttribute("position",new Zt(f,3)),this.setAttribute("normal",new Zt(p,3)),this.setAttribute("uv",new Zt(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new xc(e.radius,e.height,e.capSegments,e.radialSegments,e.heightSegments)}}class gc extends fn{constructor(e=1,t=1,n=1,a=32,o=1,l=!1,f=0,p=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:a,heightSegments:o,openEnded:l,thetaStart:f,thetaLength:p};const h=this;a=Math.floor(a),o=Math.floor(o);const _=[],g=[],v=[],b=[];let R=0;const P=[],E=n/2;let m=0;U(),l===!1&&(e>0&&L(!0),t>0&&L(!1)),this.setIndex(_),this.setAttribute("position",new Zt(g,3)),this.setAttribute("normal",new Zt(v,3)),this.setAttribute("uv",new Zt(b,2));function U(){const N=new K,z=new K;let A=0;const V=(t-e)/n;for(let Z=0;Z<=o;Z++){const D=[],y=Z/o,G=y*(t-e)+e;for(let Y=0;Y<=a;Y++){const J=Y/a,ee=J*p+f,fe=Math.sin(ee),le=Math.cos(ee);z.x=G*fe,z.y=-y*n+E,z.z=G*le,g.push(z.x,z.y,z.z),N.set(fe,V,le).normalize(),v.push(N.x,N.y,N.z),b.push(J,1-y),D.push(R++)}P.push(D)}for(let Z=0;Z<a;Z++)for(let D=0;D<o;D++){const y=P[D][Z],G=P[D+1][Z],Y=P[D+1][Z+1],J=P[D][Z+1];(e>0||D!==0)&&(_.push(y,G,J),A+=3),(t>0||D!==o-1)&&(_.push(G,Y,J),A+=3)}h.addGroup(m,A,0),m+=A}function L(N){const z=R,A=new et,V=new K;let Z=0;const D=N===!0?e:t,y=N===!0?1:-1;for(let Y=1;Y<=a;Y++)g.push(0,E*y,0),v.push(0,y,0),b.push(.5,.5),R++;const G=R;for(let Y=0;Y<=a;Y++){const ee=Y/a*p+f,fe=Math.cos(ee),le=Math.sin(ee);V.x=D*le,V.y=E*y,V.z=D*fe,g.push(V.x,V.y,V.z),v.push(0,y,0),A.x=fe*.5+.5,A.y=le*.5*y+.5,b.push(A.x,A.y),R++}for(let Y=0;Y<a;Y++){const J=z+Y,ee=G+Y;N===!0?_.push(ee,ee+1,J):_.push(ee+1,ee,J),Z+=3}h.addGroup(m,Z,N===!0?1:2),m+=Z}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new gc(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class is extends fn{constructor(e=1,t=1,n=1,a=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:a};const o=e/2,l=t/2,f=Math.floor(n),p=Math.floor(a),h=f+1,_=p+1,g=e/f,v=t/p,b=[],R=[],P=[],E=[];for(let m=0;m<_;m++){const U=m*v-l;for(let L=0;L<h;L++){const N=L*g-o;R.push(N,-U,0),P.push(0,0,1),E.push(L/f),E.push(1-m/p)}}for(let m=0;m<p;m++)for(let U=0;U<f;U++){const L=U+h*m,N=U+h*(m+1),z=U+1+h*(m+1),A=U+1+h*m;b.push(L,N,A),b.push(N,z,A)}this.setIndex(b),this.setAttribute("position",new Zt(R,3)),this.setAttribute("normal",new Zt(P,3)),this.setAttribute("uv",new Zt(E,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new is(e.width,e.height,e.widthSegments,e.heightSegments)}}class $r extends fn{constructor(e=1,t=32,n=16,a=0,o=Math.PI*2,l=0,f=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:a,phiLength:o,thetaStart:l,thetaLength:f},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const p=Math.min(l+f,Math.PI);let h=0;const _=[],g=new K,v=new K,b=[],R=[],P=[],E=[];for(let m=0;m<=n;m++){const U=[],L=m/n;let N=0;m===0&&l===0?N=.5/t:m===n&&p===Math.PI&&(N=-.5/t);for(let z=0;z<=t;z++){const A=z/t;g.x=-e*Math.cos(a+A*o)*Math.sin(l+L*f),g.y=e*Math.cos(l+L*f),g.z=e*Math.sin(a+A*o)*Math.sin(l+L*f),R.push(g.x,g.y,g.z),v.copy(g).normalize(),P.push(v.x,v.y,v.z),E.push(A+N,1-L),U.push(h++)}_.push(U)}for(let m=0;m<n;m++)for(let U=0;U<t;U++){const L=_[m][U+1],N=_[m][U],z=_[m+1][U],A=_[m+1][U+1];(m!==0||l>0)&&b.push(L,N,A),(m!==n-1||p<Math.PI)&&b.push(N,z,A)}this.setIndex(b),this.setAttribute("position",new Zt(R,3)),this.setAttribute("normal",new Zt(P,3)),this.setAttribute("uv",new Zt(E,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new $r(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class R_ extends Cr{constructor(e){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new xt(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new xt(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Du,this.normalScale=new et(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new jn,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.defines={STANDARD:""},this.color.copy(e.color),this.roughness=e.roughness,this.metalness=e.metalness,this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.roughnessMap=e.roughnessMap,this.metalnessMap=e.metalnessMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.envMapIntensity=e.envMapIntensity,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}}class C_ extends Cr{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=O0,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class P_ extends Cr{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class $u extends en{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new xt(e),this.intensity=t}dispose(){}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){const t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,this.groundColor!==void 0&&(t.object.groundColor=this.groundColor.getHex()),this.distance!==void 0&&(t.object.distance=this.distance),this.angle!==void 0&&(t.object.angle=this.angle),this.decay!==void 0&&(t.object.decay=this.decay),this.penumbra!==void 0&&(t.object.penumbra=this.penumbra),this.shadow!==void 0&&(t.object.shadow=this.shadow.toJSON()),this.target!==void 0&&(t.object.target=this.target.uuid),t}}const so=new Bt,Gl=new K,Hl=new K;class D_{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new et(512,512),this.mapType=$n,this.map=null,this.mapPass=null,this.matrix=new Bt,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new mc,this._frameExtents=new et(1,1),this._viewportCount=1,this._viewports=[new zt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){const t=this.camera,n=this.matrix;Gl.setFromMatrixPosition(e.matrixWorld),t.position.copy(Gl),Hl.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(Hl),t.updateMatrixWorld(),so.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(so,t.coordinateSystem,t.reversedDepth),t.reversedDepth?n.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(so)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){const e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}}class ju extends Vu{constructor(e=-1,t=1,n=1,a=-1,o=.1,l=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=a,this.near=o,this.far=l,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,a,o,l){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=a,this.view.width=o,this.view.height=l,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,a=(this.top+this.bottom)/2;let o=n-e,l=n+e,f=a+t,p=a-t;if(this.view!==null&&this.view.enabled){const h=(this.right-this.left)/this.view.fullWidth/this.zoom,_=(this.top-this.bottom)/this.view.fullHeight/this.zoom;o+=h*this.view.offsetX,l=o+h*this.view.width,f-=_*this.view.offsetY,p=f-_*this.view.height}this.projectionMatrix.makeOrthographic(o,l,f,p,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class L_ extends D_{constructor(){super(new ju(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}}class F_ extends $u{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(en.DEFAULT_UP),this.updateMatrix(),this.target=new en,this.shadow=new L_}dispose(){this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}}class I_ extends $u{constructor(e,t){super(e,t),this.isAmbientLight=!0,this.type="AmbientLight"}}class U_ extends Tn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class Wl{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=dt(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(dt(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}class N_ extends ji{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){Ze("Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}}function Xl(r,e,t,n){const a=O_(n);switch(t){case Ru:return r*e;case Pu:return r*e/a.components*a.byteLength;case lc:return r*e/a.components*a.byteLength;case uc:return r*e*2/a.components*a.byteLength;case fc:return r*e*2/a.components*a.byteLength;case Cu:return r*e*3/a.components*a.byteLength;case Nn:return r*e*4/a.components*a.byteLength;case dc:return r*e*4/a.components*a.byteLength;case Ks:case Zs:return Math.floor((r+3)/4)*Math.floor((e+3)/4)*8;case Js:case Qs:return Math.floor((r+3)/4)*Math.floor((e+3)/4)*16;case Ro:case Po:return Math.max(r,16)*Math.max(e,8)/4;case Ao:case Co:return Math.max(r,8)*Math.max(e,8)/2;case Do:case Lo:return Math.floor((r+3)/4)*Math.floor((e+3)/4)*8;case Fo:return Math.floor((r+3)/4)*Math.floor((e+3)/4)*16;case Io:return Math.floor((r+3)/4)*Math.floor((e+3)/4)*16;case Uo:return Math.floor((r+4)/5)*Math.floor((e+3)/4)*16;case No:return Math.floor((r+4)/5)*Math.floor((e+4)/5)*16;case Oo:return Math.floor((r+5)/6)*Math.floor((e+4)/5)*16;case ko:return Math.floor((r+5)/6)*Math.floor((e+5)/6)*16;case Bo:return Math.floor((r+7)/8)*Math.floor((e+4)/5)*16;case zo:return Math.floor((r+7)/8)*Math.floor((e+5)/6)*16;case Vo:return Math.floor((r+7)/8)*Math.floor((e+7)/8)*16;case Go:return Math.floor((r+9)/10)*Math.floor((e+4)/5)*16;case Ho:return Math.floor((r+9)/10)*Math.floor((e+5)/6)*16;case Wo:return Math.floor((r+9)/10)*Math.floor((e+7)/8)*16;case Xo:return Math.floor((r+9)/10)*Math.floor((e+9)/10)*16;case $o:return Math.floor((r+11)/12)*Math.floor((e+9)/10)*16;case jo:return Math.floor((r+11)/12)*Math.floor((e+11)/12)*16;case Yo:case qo:case Ko:return Math.ceil(r/4)*Math.ceil(e/4)*16;case Zo:case Jo:return Math.ceil(r/4)*Math.ceil(e/4)*8;case Qo:case ec:return Math.ceil(r/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function O_(r){switch(r){case $n:case Mu:return{byteLength:1,components:1};case jr:case Tu:case Rr:return{byteLength:2,components:1};case oc:case cc:return{byteLength:2,components:4};case Wi:case ac:case ii:return{byteLength:4,components:1};case wu:case Au:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${r}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:sc}}));typeof window<"u"&&(window.__THREE__?Ze("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=sc);/**
 * @license
 * Copyright 2010-2025 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function Yu(){let r=null,e=!1,t=null,n=null;function a(o,l){t(o,l),n=r.requestAnimationFrame(a)}return{start:function(){e!==!0&&t!==null&&(n=r.requestAnimationFrame(a),e=!0)},stop:function(){r.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(o){t=o},setContext:function(o){r=o}}}function k_(r){const e=new WeakMap;function t(f,p){const h=f.array,_=f.usage,g=h.byteLength,v=r.createBuffer();r.bindBuffer(p,v),r.bufferData(p,h,_),f.onUploadCallback();let b;if(h instanceof Float32Array)b=r.FLOAT;else if(typeof Float16Array<"u"&&h instanceof Float16Array)b=r.HALF_FLOAT;else if(h instanceof Uint16Array)f.isFloat16BufferAttribute?b=r.HALF_FLOAT:b=r.UNSIGNED_SHORT;else if(h instanceof Int16Array)b=r.SHORT;else if(h instanceof Uint32Array)b=r.UNSIGNED_INT;else if(h instanceof Int32Array)b=r.INT;else if(h instanceof Int8Array)b=r.BYTE;else if(h instanceof Uint8Array)b=r.UNSIGNED_BYTE;else if(h instanceof Uint8ClampedArray)b=r.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+h);return{buffer:v,type:b,bytesPerElement:h.BYTES_PER_ELEMENT,version:f.version,size:g}}function n(f,p,h){const _=p.array,g=p.updateRanges;if(r.bindBuffer(h,f),g.length===0)r.bufferSubData(h,0,_);else{g.sort((b,R)=>b.start-R.start);let v=0;for(let b=1;b<g.length;b++){const R=g[v],P=g[b];P.start<=R.start+R.count+1?R.count=Math.max(R.count,P.start+P.count-R.start):(++v,g[v]=P)}g.length=v+1;for(let b=0,R=g.length;b<R;b++){const P=g[b];r.bufferSubData(h,P.start*_.BYTES_PER_ELEMENT,_,P.start,P.count)}p.clearUpdateRanges()}p.onUploadCallback()}function a(f){return f.isInterleavedBufferAttribute&&(f=f.data),e.get(f)}function o(f){f.isInterleavedBufferAttribute&&(f=f.data);const p=e.get(f);p&&(r.deleteBuffer(p.buffer),e.delete(f))}function l(f,p){if(f.isInterleavedBufferAttribute&&(f=f.data),f.isGLBufferAttribute){const _=e.get(f);(!_||_.version<f.version)&&e.set(f,{buffer:f.buffer,type:f.type,bytesPerElement:f.elementSize,version:f.version});return}const h=e.get(f);if(h===void 0)e.set(f,t(f,p));else if(h.version<f.version){if(h.size!==f.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(h.buffer,f,p),h.version=f.version}}return{get:a,remove:o,update:l}}var B_=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,z_=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,V_=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,G_=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,H_=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,W_=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,X_=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,$_=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,j_=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Y_=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,q_=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,K_=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Z_=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,J_=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Q_=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,ex=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,tx=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,nx=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,ix=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,rx=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,sx=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,ax=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,ox=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,cx=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,lx=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,ux=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,fx=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,dx=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,hx=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,px=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,mx="gl_FragColor = linearToOutputTexel( gl_FragColor );",_x=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,xx=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,gx=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,vx=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,yx=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,Sx=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,bx=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,Ex=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,Mx=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Tx=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,wx=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,Ax=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,Rx=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,Cx=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Px=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,Dx=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,Lx=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Fx=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Ix=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Ux=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Nx=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,Ox=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 uv = vec2( roughness, dotNV );
	return texture2D( dfgLUT, uv ).rg;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = DFGApprox( vec3(0.0, 0.0, 1.0), vec3(sqrt(1.0 - dotNV * dotNV), 0.0, dotNV), material.roughness );
	vec2 dfgL = DFGApprox( vec3(0.0, 0.0, 1.0), vec3(sqrt(1.0 - dotNL * dotNL), 0.0, dotNL), material.roughness );
	vec3 FssEss_V = material.specularColor * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColor * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColor + ( 1.0 - material.specularColor ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,kx=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,Bx=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,zx=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Vx=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Gx=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Hx=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Wx=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Xx=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,$x=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,jx=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Yx=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,qx=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Kx=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Zx=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Jx=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Qx=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,eg=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,tg=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,ng=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,ig=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,rg=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,sg=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,ag=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,og=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,cg=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,lg=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,ug=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,fg=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,dg=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,hg=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,pg=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,mg=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,_g=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,xg=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,gg=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,vg=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,yg=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		float depth = unpackRGBAToDepth( texture2D( depths, uv ) );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			return step( depth, compare );
		#else
			return step( compare, depth );
		#endif
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow( sampler2D shadow, vec2 uv, float compare ) {
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			float hard_shadow = step( distribution.x, compare );
		#else
			float hard_shadow = step( compare, distribution.x );
		#endif
		if ( hard_shadow != 1.0 ) {
			float distance = compare - distribution.x;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,Sg=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,bg=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,Eg=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,Mg=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,Tg=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,wg=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,Ag=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,Rg=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,Cg=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,Pg=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,Dg=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,Lg=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,Fg=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Ig=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Ug=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Ng=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Og=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const kg=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Bg=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,zg=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Vg=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Gg=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Hg=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Wg=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Xg=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,$g=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,jg=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Yg=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,qg=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Kg=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Zg=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Jg=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Qg=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ev=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,tv=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,nv=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,iv=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,rv=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,sv=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,av=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,ov=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,cv=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,lv=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,uv=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,fv=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,dv=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,hv=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,pv=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,mv=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,_v=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,xv=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,st={alphahash_fragment:B_,alphahash_pars_fragment:z_,alphamap_fragment:V_,alphamap_pars_fragment:G_,alphatest_fragment:H_,alphatest_pars_fragment:W_,aomap_fragment:X_,aomap_pars_fragment:$_,batching_pars_vertex:j_,batching_vertex:Y_,begin_vertex:q_,beginnormal_vertex:K_,bsdfs:Z_,iridescence_fragment:J_,bumpmap_pars_fragment:Q_,clipping_planes_fragment:ex,clipping_planes_pars_fragment:tx,clipping_planes_pars_vertex:nx,clipping_planes_vertex:ix,color_fragment:rx,color_pars_fragment:sx,color_pars_vertex:ax,color_vertex:ox,common:cx,cube_uv_reflection_fragment:lx,defaultnormal_vertex:ux,displacementmap_pars_vertex:fx,displacementmap_vertex:dx,emissivemap_fragment:hx,emissivemap_pars_fragment:px,colorspace_fragment:mx,colorspace_pars_fragment:_x,envmap_fragment:xx,envmap_common_pars_fragment:gx,envmap_pars_fragment:vx,envmap_pars_vertex:yx,envmap_physical_pars_fragment:Dx,envmap_vertex:Sx,fog_vertex:bx,fog_pars_vertex:Ex,fog_fragment:Mx,fog_pars_fragment:Tx,gradientmap_pars_fragment:wx,lightmap_pars_fragment:Ax,lights_lambert_fragment:Rx,lights_lambert_pars_fragment:Cx,lights_pars_begin:Px,lights_toon_fragment:Lx,lights_toon_pars_fragment:Fx,lights_phong_fragment:Ix,lights_phong_pars_fragment:Ux,lights_physical_fragment:Nx,lights_physical_pars_fragment:Ox,lights_fragment_begin:kx,lights_fragment_maps:Bx,lights_fragment_end:zx,logdepthbuf_fragment:Vx,logdepthbuf_pars_fragment:Gx,logdepthbuf_pars_vertex:Hx,logdepthbuf_vertex:Wx,map_fragment:Xx,map_pars_fragment:$x,map_particle_fragment:jx,map_particle_pars_fragment:Yx,metalnessmap_fragment:qx,metalnessmap_pars_fragment:Kx,morphinstance_vertex:Zx,morphcolor_vertex:Jx,morphnormal_vertex:Qx,morphtarget_pars_vertex:eg,morphtarget_vertex:tg,normal_fragment_begin:ng,normal_fragment_maps:ig,normal_pars_fragment:rg,normal_pars_vertex:sg,normal_vertex:ag,normalmap_pars_fragment:og,clearcoat_normal_fragment_begin:cg,clearcoat_normal_fragment_maps:lg,clearcoat_pars_fragment:ug,iridescence_pars_fragment:fg,opaque_fragment:dg,packing:hg,premultiplied_alpha_fragment:pg,project_vertex:mg,dithering_fragment:_g,dithering_pars_fragment:xg,roughnessmap_fragment:gg,roughnessmap_pars_fragment:vg,shadowmap_pars_fragment:yg,shadowmap_pars_vertex:Sg,shadowmap_vertex:bg,shadowmask_pars_fragment:Eg,skinbase_vertex:Mg,skinning_pars_vertex:Tg,skinning_vertex:wg,skinnormal_vertex:Ag,specularmap_fragment:Rg,specularmap_pars_fragment:Cg,tonemapping_fragment:Pg,tonemapping_pars_fragment:Dg,transmission_fragment:Lg,transmission_pars_fragment:Fg,uv_pars_fragment:Ig,uv_pars_vertex:Ug,uv_vertex:Ng,worldpos_vertex:Og,background_vert:kg,background_frag:Bg,backgroundCube_vert:zg,backgroundCube_frag:Vg,cube_vert:Gg,cube_frag:Hg,depth_vert:Wg,depth_frag:Xg,distanceRGBA_vert:$g,distanceRGBA_frag:jg,equirect_vert:Yg,equirect_frag:qg,linedashed_vert:Kg,linedashed_frag:Zg,meshbasic_vert:Jg,meshbasic_frag:Qg,meshlambert_vert:ev,meshlambert_frag:tv,meshmatcap_vert:nv,meshmatcap_frag:iv,meshnormal_vert:rv,meshnormal_frag:sv,meshphong_vert:av,meshphong_frag:ov,meshphysical_vert:cv,meshphysical_frag:lv,meshtoon_vert:uv,meshtoon_frag:fv,points_vert:dv,points_frag:hv,shadow_vert:pv,shadow_frag:mv,sprite_vert:_v,sprite_frag:xv},Ce={common:{diffuse:{value:new xt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new tt},alphaMap:{value:null},alphaMapTransform:{value:new tt},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new tt}},envmap:{envMap:{value:null},envMapRotation:{value:new tt},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new tt}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new tt}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new tt},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new tt},normalScale:{value:new et(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new tt},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new tt}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new tt}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new tt}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new xt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new xt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new tt},alphaTest:{value:0},uvTransform:{value:new tt}},sprite:{diffuse:{value:new xt(16777215)},opacity:{value:1},center:{value:new et(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new tt},alphaMap:{value:null},alphaMapTransform:{value:new tt},alphaTest:{value:0}}},Gn={basic:{uniforms:ln([Ce.common,Ce.specularmap,Ce.envmap,Ce.aomap,Ce.lightmap,Ce.fog]),vertexShader:st.meshbasic_vert,fragmentShader:st.meshbasic_frag},lambert:{uniforms:ln([Ce.common,Ce.specularmap,Ce.envmap,Ce.aomap,Ce.lightmap,Ce.emissivemap,Ce.bumpmap,Ce.normalmap,Ce.displacementmap,Ce.fog,Ce.lights,{emissive:{value:new xt(0)}}]),vertexShader:st.meshlambert_vert,fragmentShader:st.meshlambert_frag},phong:{uniforms:ln([Ce.common,Ce.specularmap,Ce.envmap,Ce.aomap,Ce.lightmap,Ce.emissivemap,Ce.bumpmap,Ce.normalmap,Ce.displacementmap,Ce.fog,Ce.lights,{emissive:{value:new xt(0)},specular:{value:new xt(1118481)},shininess:{value:30}}]),vertexShader:st.meshphong_vert,fragmentShader:st.meshphong_frag},standard:{uniforms:ln([Ce.common,Ce.envmap,Ce.aomap,Ce.lightmap,Ce.emissivemap,Ce.bumpmap,Ce.normalmap,Ce.displacementmap,Ce.roughnessmap,Ce.metalnessmap,Ce.fog,Ce.lights,{emissive:{value:new xt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:st.meshphysical_vert,fragmentShader:st.meshphysical_frag},toon:{uniforms:ln([Ce.common,Ce.aomap,Ce.lightmap,Ce.emissivemap,Ce.bumpmap,Ce.normalmap,Ce.displacementmap,Ce.gradientmap,Ce.fog,Ce.lights,{emissive:{value:new xt(0)}}]),vertexShader:st.meshtoon_vert,fragmentShader:st.meshtoon_frag},matcap:{uniforms:ln([Ce.common,Ce.bumpmap,Ce.normalmap,Ce.displacementmap,Ce.fog,{matcap:{value:null}}]),vertexShader:st.meshmatcap_vert,fragmentShader:st.meshmatcap_frag},points:{uniforms:ln([Ce.points,Ce.fog]),vertexShader:st.points_vert,fragmentShader:st.points_frag},dashed:{uniforms:ln([Ce.common,Ce.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:st.linedashed_vert,fragmentShader:st.linedashed_frag},depth:{uniforms:ln([Ce.common,Ce.displacementmap]),vertexShader:st.depth_vert,fragmentShader:st.depth_frag},normal:{uniforms:ln([Ce.common,Ce.bumpmap,Ce.normalmap,Ce.displacementmap,{opacity:{value:1}}]),vertexShader:st.meshnormal_vert,fragmentShader:st.meshnormal_frag},sprite:{uniforms:ln([Ce.sprite,Ce.fog]),vertexShader:st.sprite_vert,fragmentShader:st.sprite_frag},background:{uniforms:{uvTransform:{value:new tt},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:st.background_vert,fragmentShader:st.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new tt}},vertexShader:st.backgroundCube_vert,fragmentShader:st.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:st.cube_vert,fragmentShader:st.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:st.equirect_vert,fragmentShader:st.equirect_frag},distanceRGBA:{uniforms:ln([Ce.common,Ce.displacementmap,{referencePosition:{value:new K},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:st.distanceRGBA_vert,fragmentShader:st.distanceRGBA_frag},shadow:{uniforms:ln([Ce.lights,Ce.fog,{color:{value:new xt(0)},opacity:{value:1}}]),vertexShader:st.shadow_vert,fragmentShader:st.shadow_frag}};Gn.physical={uniforms:ln([Gn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new tt},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new tt},clearcoatNormalScale:{value:new et(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new tt},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new tt},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new tt},sheen:{value:0},sheenColor:{value:new xt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new tt},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new tt},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new tt},transmissionSamplerSize:{value:new et},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new tt},attenuationDistance:{value:0},attenuationColor:{value:new xt(0)},specularColor:{value:new xt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new tt},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new tt},anisotropyVector:{value:new et},anisotropyMap:{value:null},anisotropyMapTransform:{value:new tt}}]),vertexShader:st.meshphysical_vert,fragmentShader:st.meshphysical_frag};const $s={r:0,b:0,g:0},ki=new jn,gv=new Bt;function vv(r,e,t,n,a,o,l){const f=new xt(0);let p=o===!0?0:1,h,_,g=null,v=0,b=null;function R(L){let N=L.isScene===!0?L.background:null;return N&&N.isTexture&&(N=(L.backgroundBlurriness>0?t:e).get(N)),N}function P(L){let N=!1;const z=R(L);z===null?m(f,p):z&&z.isColor&&(m(z,1),N=!0);const A=r.xr.getEnvironmentBlendMode();A==="additive"?n.buffers.color.setClear(0,0,0,1,l):A==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,l),(r.autoClear||N)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),r.clear(r.autoClearColor,r.autoClearDepth,r.autoClearStencil))}function E(L,N){const z=R(N);z&&(z.isCubeTexture||z.mapping===ua)?(_===void 0&&(_=new li(new Pr(1,1,1),new ui({name:"BackgroundCubeMaterial",uniforms:wr(Gn.backgroundCube.uniforms),vertexShader:Gn.backgroundCube.vertexShader,fragmentShader:Gn.backgroundCube.fragmentShader,side:_n,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),_.geometry.deleteAttribute("normal"),_.geometry.deleteAttribute("uv"),_.onBeforeRender=function(A,V,Z){this.matrixWorld.copyPosition(Z.matrixWorld)},Object.defineProperty(_.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),a.update(_)),ki.copy(N.backgroundRotation),ki.x*=-1,ki.y*=-1,ki.z*=-1,z.isCubeTexture&&z.isRenderTargetTexture===!1&&(ki.y*=-1,ki.z*=-1),_.material.uniforms.envMap.value=z,_.material.uniforms.flipEnvMap.value=z.isCubeTexture&&z.isRenderTargetTexture===!1?-1:1,_.material.uniforms.backgroundBlurriness.value=N.backgroundBlurriness,_.material.uniforms.backgroundIntensity.value=N.backgroundIntensity,_.material.uniforms.backgroundRotation.value.setFromMatrix4(gv.makeRotationFromEuler(ki)),_.material.toneMapped=yt.getTransfer(z.colorSpace)!==Ft,(g!==z||v!==z.version||b!==r.toneMapping)&&(_.material.needsUpdate=!0,g=z,v=z.version,b=r.toneMapping),_.layers.enableAll(),L.unshift(_,_.geometry,_.material,0,0,null)):z&&z.isTexture&&(h===void 0&&(h=new li(new is(2,2),new ui({name:"BackgroundMaterial",uniforms:wr(Gn.background.uniforms),vertexShader:Gn.background.vertexShader,fragmentShader:Gn.background.fragmentShader,side:Ei,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),h.geometry.deleteAttribute("normal"),Object.defineProperty(h.material,"map",{get:function(){return this.uniforms.t2D.value}}),a.update(h)),h.material.uniforms.t2D.value=z,h.material.uniforms.backgroundIntensity.value=N.backgroundIntensity,h.material.toneMapped=yt.getTransfer(z.colorSpace)!==Ft,z.matrixAutoUpdate===!0&&z.updateMatrix(),h.material.uniforms.uvTransform.value.copy(z.matrix),(g!==z||v!==z.version||b!==r.toneMapping)&&(h.material.needsUpdate=!0,g=z,v=z.version,b=r.toneMapping),h.layers.enableAll(),L.unshift(h,h.geometry,h.material,0,0,null))}function m(L,N){L.getRGB($s,zu(r)),n.buffers.color.setClear($s.r,$s.g,$s.b,N,l)}function U(){_!==void 0&&(_.geometry.dispose(),_.material.dispose(),_=void 0),h!==void 0&&(h.geometry.dispose(),h.material.dispose(),h=void 0)}return{getClearColor:function(){return f},setClearColor:function(L,N=1){f.set(L),p=N,m(f,p)},getClearAlpha:function(){return p},setClearAlpha:function(L){p=L,m(f,p)},render:P,addToRenderList:E,dispose:U}}function yv(r,e){const t=r.getParameter(r.MAX_VERTEX_ATTRIBS),n={},a=v(null);let o=a,l=!1;function f(y,G,Y,J,ee){let fe=!1;const le=g(J,Y,G);o!==le&&(o=le,h(o.object)),fe=b(y,J,Y,ee),fe&&R(y,J,Y,ee),ee!==null&&e.update(ee,r.ELEMENT_ARRAY_BUFFER),(fe||l)&&(l=!1,N(y,G,Y,J),ee!==null&&r.bindBuffer(r.ELEMENT_ARRAY_BUFFER,e.get(ee).buffer))}function p(){return r.createVertexArray()}function h(y){return r.bindVertexArray(y)}function _(y){return r.deleteVertexArray(y)}function g(y,G,Y){const J=Y.wireframe===!0;let ee=n[y.id];ee===void 0&&(ee={},n[y.id]=ee);let fe=ee[G.id];fe===void 0&&(fe={},ee[G.id]=fe);let le=fe[J];return le===void 0&&(le=v(p()),fe[J]=le),le}function v(y){const G=[],Y=[],J=[];for(let ee=0;ee<t;ee++)G[ee]=0,Y[ee]=0,J[ee]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:G,enabledAttributes:Y,attributeDivisors:J,object:y,attributes:{},index:null}}function b(y,G,Y,J){const ee=o.attributes,fe=G.attributes;let le=0;const Ee=Y.getAttributes();for(const se in Ee)if(Ee[se].location>=0){const we=ee[se];let Ge=fe[se];if(Ge===void 0&&(se==="instanceMatrix"&&y.instanceMatrix&&(Ge=y.instanceMatrix),se==="instanceColor"&&y.instanceColor&&(Ge=y.instanceColor)),we===void 0||we.attribute!==Ge||Ge&&we.data!==Ge.data)return!0;le++}return o.attributesNum!==le||o.index!==J}function R(y,G,Y,J){const ee={},fe=G.attributes;let le=0;const Ee=Y.getAttributes();for(const se in Ee)if(Ee[se].location>=0){let we=fe[se];we===void 0&&(se==="instanceMatrix"&&y.instanceMatrix&&(we=y.instanceMatrix),se==="instanceColor"&&y.instanceColor&&(we=y.instanceColor));const Ge={};Ge.attribute=we,we&&we.data&&(Ge.data=we.data),ee[se]=Ge,le++}o.attributes=ee,o.attributesNum=le,o.index=J}function P(){const y=o.newAttributes;for(let G=0,Y=y.length;G<Y;G++)y[G]=0}function E(y){m(y,0)}function m(y,G){const Y=o.newAttributes,J=o.enabledAttributes,ee=o.attributeDivisors;Y[y]=1,J[y]===0&&(r.enableVertexAttribArray(y),J[y]=1),ee[y]!==G&&(r.vertexAttribDivisor(y,G),ee[y]=G)}function U(){const y=o.newAttributes,G=o.enabledAttributes;for(let Y=0,J=G.length;Y<J;Y++)G[Y]!==y[Y]&&(r.disableVertexAttribArray(Y),G[Y]=0)}function L(y,G,Y,J,ee,fe,le){le===!0?r.vertexAttribIPointer(y,G,Y,ee,fe):r.vertexAttribPointer(y,G,Y,J,ee,fe)}function N(y,G,Y,J){P();const ee=J.attributes,fe=Y.getAttributes(),le=G.defaultAttributeValues;for(const Ee in fe){const se=fe[Ee];if(se.location>=0){let Me=ee[Ee];if(Me===void 0&&(Ee==="instanceMatrix"&&y.instanceMatrix&&(Me=y.instanceMatrix),Ee==="instanceColor"&&y.instanceColor&&(Me=y.instanceColor)),Me!==void 0){const we=Me.normalized,Ge=Me.itemSize,ot=e.get(Me);if(ot===void 0)continue;const _t=ot.buffer,Je=ot.type,at=ot.bytesPerElement,ce=Je===r.INT||Je===r.UNSIGNED_INT||Me.gpuType===ac;if(Me.isInterleavedBufferAttribute){const de=Me.data,ie=de.stride,ye=Me.offset;if(de.isInstancedInterleavedBuffer){for(let Be=0;Be<se.locationSize;Be++)m(se.location+Be,de.meshPerAttribute);y.isInstancedMesh!==!0&&J._maxInstanceCount===void 0&&(J._maxInstanceCount=de.meshPerAttribute*de.count)}else for(let Be=0;Be<se.locationSize;Be++)E(se.location+Be);r.bindBuffer(r.ARRAY_BUFFER,_t);for(let Be=0;Be<se.locationSize;Be++)L(se.location+Be,Ge/se.locationSize,Je,we,ie*at,(ye+Ge/se.locationSize*Be)*at,ce)}else{if(Me.isInstancedBufferAttribute){for(let de=0;de<se.locationSize;de++)m(se.location+de,Me.meshPerAttribute);y.isInstancedMesh!==!0&&J._maxInstanceCount===void 0&&(J._maxInstanceCount=Me.meshPerAttribute*Me.count)}else for(let de=0;de<se.locationSize;de++)E(se.location+de);r.bindBuffer(r.ARRAY_BUFFER,_t);for(let de=0;de<se.locationSize;de++)L(se.location+de,Ge/se.locationSize,Je,we,Ge*at,Ge/se.locationSize*de*at,ce)}}else if(le!==void 0){const we=le[Ee];if(we!==void 0)switch(we.length){case 2:r.vertexAttrib2fv(se.location,we);break;case 3:r.vertexAttrib3fv(se.location,we);break;case 4:r.vertexAttrib4fv(se.location,we);break;default:r.vertexAttrib1fv(se.location,we)}}}}U()}function z(){Z();for(const y in n){const G=n[y];for(const Y in G){const J=G[Y];for(const ee in J)_(J[ee].object),delete J[ee];delete G[Y]}delete n[y]}}function A(y){if(n[y.id]===void 0)return;const G=n[y.id];for(const Y in G){const J=G[Y];for(const ee in J)_(J[ee].object),delete J[ee];delete G[Y]}delete n[y.id]}function V(y){for(const G in n){const Y=n[G];if(Y[y.id]===void 0)continue;const J=Y[y.id];for(const ee in J)_(J[ee].object),delete J[ee];delete Y[y.id]}}function Z(){D(),l=!0,o!==a&&(o=a,h(o.object))}function D(){a.geometry=null,a.program=null,a.wireframe=!1}return{setup:f,reset:Z,resetDefaultState:D,dispose:z,releaseStatesOfGeometry:A,releaseStatesOfProgram:V,initAttributes:P,enableAttribute:E,disableUnusedAttributes:U}}function Sv(r,e,t){let n;function a(h){n=h}function o(h,_){r.drawArrays(n,h,_),t.update(_,n,1)}function l(h,_,g){g!==0&&(r.drawArraysInstanced(n,h,_,g),t.update(_,n,g))}function f(h,_,g){if(g===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,h,0,_,0,g);let b=0;for(let R=0;R<g;R++)b+=_[R];t.update(b,n,1)}function p(h,_,g,v){if(g===0)return;const b=e.get("WEBGL_multi_draw");if(b===null)for(let R=0;R<h.length;R++)l(h[R],_[R],v[R]);else{b.multiDrawArraysInstancedWEBGL(n,h,0,_,0,v,0,g);let R=0;for(let P=0;P<g;P++)R+=_[P]*v[P];t.update(R,n,1)}}this.setMode=a,this.render=o,this.renderInstances=l,this.renderMultiDraw=f,this.renderMultiDrawInstances=p}function bv(r,e,t,n){let a;function o(){if(a!==void 0)return a;if(e.has("EXT_texture_filter_anisotropic")===!0){const V=e.get("EXT_texture_filter_anisotropic");a=r.getParameter(V.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else a=0;return a}function l(V){return!(V!==Nn&&n.convert(V)!==r.getParameter(r.IMPLEMENTATION_COLOR_READ_FORMAT))}function f(V){const Z=V===Rr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(V!==$n&&n.convert(V)!==r.getParameter(r.IMPLEMENTATION_COLOR_READ_TYPE)&&V!==ii&&!Z)}function p(V){if(V==="highp"){if(r.getShaderPrecisionFormat(r.VERTEX_SHADER,r.HIGH_FLOAT).precision>0&&r.getShaderPrecisionFormat(r.FRAGMENT_SHADER,r.HIGH_FLOAT).precision>0)return"highp";V="mediump"}return V==="mediump"&&r.getShaderPrecisionFormat(r.VERTEX_SHADER,r.MEDIUM_FLOAT).precision>0&&r.getShaderPrecisionFormat(r.FRAGMENT_SHADER,r.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let h=t.precision!==void 0?t.precision:"highp";const _=p(h);_!==h&&(Ze("WebGLRenderer:",h,"not supported, using",_,"instead."),h=_);const g=t.logarithmicDepthBuffer===!0,v=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),b=r.getParameter(r.MAX_TEXTURE_IMAGE_UNITS),R=r.getParameter(r.MAX_VERTEX_TEXTURE_IMAGE_UNITS),P=r.getParameter(r.MAX_TEXTURE_SIZE),E=r.getParameter(r.MAX_CUBE_MAP_TEXTURE_SIZE),m=r.getParameter(r.MAX_VERTEX_ATTRIBS),U=r.getParameter(r.MAX_VERTEX_UNIFORM_VECTORS),L=r.getParameter(r.MAX_VARYING_VECTORS),N=r.getParameter(r.MAX_FRAGMENT_UNIFORM_VECTORS),z=R>0,A=r.getParameter(r.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:o,getMaxPrecision:p,textureFormatReadable:l,textureTypeReadable:f,precision:h,logarithmicDepthBuffer:g,reversedDepthBuffer:v,maxTextures:b,maxVertexTextures:R,maxTextureSize:P,maxCubemapSize:E,maxAttributes:m,maxVertexUniforms:U,maxVaryings:L,maxFragmentUniforms:N,vertexTextures:z,maxSamples:A}}function Ev(r){const e=this;let t=null,n=0,a=!1,o=!1;const l=new vi,f=new tt,p={value:null,needsUpdate:!1};this.uniform=p,this.numPlanes=0,this.numIntersection=0,this.init=function(g,v){const b=g.length!==0||v||n!==0||a;return a=v,n=g.length,b},this.beginShadows=function(){o=!0,_(null)},this.endShadows=function(){o=!1},this.setGlobalState=function(g,v){t=_(g,v,0)},this.setState=function(g,v,b){const R=g.clippingPlanes,P=g.clipIntersection,E=g.clipShadows,m=r.get(g);if(!a||R===null||R.length===0||o&&!E)o?_(null):h();else{const U=o?0:n,L=U*4;let N=m.clippingState||null;p.value=N,N=_(R,v,L,b);for(let z=0;z!==L;++z)N[z]=t[z];m.clippingState=N,this.numIntersection=P?this.numPlanes:0,this.numPlanes+=U}};function h(){p.value!==t&&(p.value=t,p.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function _(g,v,b,R){const P=g!==null?g.length:0;let E=null;if(P!==0){if(E=p.value,R!==!0||E===null){const m=b+P*4,U=v.matrixWorldInverse;f.getNormalMatrix(U),(E===null||E.length<m)&&(E=new Float32Array(m));for(let L=0,N=b;L!==P;++L,N+=4)l.copy(g[L]).applyMatrix4(U,f),l.normal.toArray(E,N),E[N+3]=l.constant}p.value=E,p.needsUpdate=!0}return e.numPlanes=P,e.numIntersection=0,E}}function Mv(r){let e=new WeakMap;function t(l,f){return f===Eo?l.mapping=Er:f===Mo&&(l.mapping=Mr),l}function n(l){if(l&&l.isTexture){const f=l.mapping;if(f===Eo||f===Mo)if(e.has(l)){const p=e.get(l).texture;return t(p,l.mapping)}else{const p=l.image;if(p&&p.height>0){const h=new y_(p.height);return h.fromEquirectangularTexture(r,l),e.set(l,h),l.addEventListener("dispose",a),t(h.texture,l.mapping)}else return null}}return l}function a(l){const f=l.target;f.removeEventListener("dispose",a);const p=e.get(f);p!==void 0&&(e.delete(f),p.dispose())}function o(){e=new WeakMap}return{get:n,dispose:o}}const Si=4,$l=[.125,.215,.35,.446,.526,.582],Gi=20,Tv=512,Wr=new ju,jl=new xt;let ao=null,oo=0,co=0,lo=!1;const wv=new K;class Yl{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,a=100,o={}){const{size:l=256,position:f=wv}=o;ao=this._renderer.getRenderTarget(),oo=this._renderer.getActiveCubeFace(),co=this._renderer.getActiveMipmapLevel(),lo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(l);const p=this._allocateTargets();return p.depthBuffer=!0,this._sceneToCubeUV(e,n,a,p,f),t>0&&this._blur(p,0,0,t),this._applyPMREM(p),this._cleanup(p),p}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Zl(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Kl(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(ao,oo,co),this._renderer.xr.enabled=lo,e.scissorTest=!1,ur(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===Er||e.mapping===Mr?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),ao=this._renderer.getRenderTarget(),oo=this._renderer.getActiveCubeFace(),co=this._renderer.getActiveMipmapLevel(),lo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:wn,minFilter:wn,generateMipmaps:!1,type:Rr,format:Nn,colorSpace:Tr,depthBuffer:!1},a=ql(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=ql(e,t,n);const{_lodMax:o}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=Av(o)),this._blurMaterial=Cv(o,e,t)}return a}_compileMaterial(e){const t=new li(new fn,e);this._renderer.compile(t,Wr)}_sceneToCubeUV(e,t,n,a,o){const p=new Tn(90,1,t,n),h=[1,-1,1,1,1,1],_=[1,1,1,-1,-1,-1],g=this._renderer,v=g.autoClear,b=g.toneMapping;g.getClearColor(jl),g.toneMapping=bi,g.autoClear=!1,g.state.buffers.depth.getReversed()&&(g.setRenderTarget(a),g.clearDepth(),g.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new li(new Pr,new Ou({name:"PMREM.Background",side:_n,depthWrite:!1,depthTest:!1})));const P=this._backgroundBox,E=P.material;let m=!1;const U=e.background;U?U.isColor&&(E.color.copy(U),e.background=null,m=!0):(E.color.copy(jl),m=!0);for(let L=0;L<6;L++){const N=L%3;N===0?(p.up.set(0,h[L],0),p.position.set(o.x,o.y,o.z),p.lookAt(o.x+_[L],o.y,o.z)):N===1?(p.up.set(0,0,h[L]),p.position.set(o.x,o.y,o.z),p.lookAt(o.x,o.y+_[L],o.z)):(p.up.set(0,h[L],0),p.position.set(o.x,o.y,o.z),p.lookAt(o.x,o.y,o.z+_[L]));const z=this._cubeSize;ur(a,N*z,L>2?z:0,z,z),g.setRenderTarget(a),m&&g.render(P,p),g.render(e,p)}g.toneMapping=b,g.autoClear=v,e.background=U}_textureToCubeUV(e,t){const n=this._renderer,a=e.mapping===Er||e.mapping===Mr;a?(this._cubemapMaterial===null&&(this._cubemapMaterial=Zl()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Kl());const o=a?this._cubemapMaterial:this._equirectMaterial,l=this._lodMeshes[0];l.material=o;const f=o.uniforms;f.envMap.value=e;const p=this._cubeSize;ur(t,0,0,3*p,2*p),n.setRenderTarget(t),n.render(l,Wr)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const a=this._lodMeshes.length;for(let o=1;o<a;o++)this._applyGGXFilter(e,o-1,o);t.autoClear=n}_applyGGXFilter(e,t,n){const a=this._renderer,o=this._pingPongRenderTarget;if(this._ggxMaterial===null){const U=3*Math.max(this._cubeSize,16),L=4*this._cubeSize;this._ggxMaterial=Rv(this._lodMax,U,L)}const l=this._ggxMaterial,f=this._lodMeshes[n];f.material=l;const p=l.uniforms,h=n/(this._lodMeshes.length-1),_=t/(this._lodMeshes.length-1),g=Math.sqrt(h*h-_*_),v=.05+h*.95,b=g*v,{_lodMax:R}=this,P=this._sizeLods[n],E=3*P*(n>R-Si?n-R+Si:0),m=4*(this._cubeSize-P);p.envMap.value=e.texture,p.roughness.value=b,p.mipInt.value=R-t,ur(o,E,m,3*P,2*P),a.setRenderTarget(o),a.render(f,Wr),p.envMap.value=o.texture,p.roughness.value=0,p.mipInt.value=R-n,ur(e,E,m,3*P,2*P),a.setRenderTarget(e),a.render(f,Wr)}_blur(e,t,n,a,o){const l=this._pingPongRenderTarget;this._halfBlur(e,l,t,n,a,"latitudinal",o),this._halfBlur(l,e,n,n,a,"longitudinal",o)}_halfBlur(e,t,n,a,o,l,f){const p=this._renderer,h=this._blurMaterial;l!=="latitudinal"&&l!=="longitudinal"&&Gt("blur direction must be either latitudinal or longitudinal!");const _=3,g=this._lodMeshes[a];g.material=h;const v=h.uniforms,b=this._sizeLods[n]-1,R=isFinite(o)?Math.PI/(2*b):2*Math.PI/(2*Gi-1),P=o/R,E=isFinite(o)?1+Math.floor(_*P):Gi;E>Gi&&Ze(`sigmaRadians, ${o}, is too large and will clip, as it requested ${E} samples when the maximum is set to ${Gi}`);const m=[];let U=0;for(let V=0;V<Gi;++V){const Z=V/P,D=Math.exp(-Z*Z/2);m.push(D),V===0?U+=D:V<E&&(U+=2*D)}for(let V=0;V<m.length;V++)m[V]=m[V]/U;v.envMap.value=e.texture,v.samples.value=E,v.weights.value=m,v.latitudinal.value=l==="latitudinal",f&&(v.poleAxis.value=f);const{_lodMax:L}=this;v.dTheta.value=R,v.mipInt.value=L-n;const N=this._sizeLods[a],z=3*N*(a>L-Si?a-L+Si:0),A=4*(this._cubeSize-N);ur(t,z,A,3*N,2*N),p.setRenderTarget(t),p.render(g,Wr)}}function Av(r){const e=[],t=[],n=[];let a=r;const o=r-Si+1+$l.length;for(let l=0;l<o;l++){const f=Math.pow(2,a);e.push(f);let p=1/f;l>r-Si?p=$l[l-r+Si-1]:l===0&&(p=0),t.push(p);const h=1/(f-2),_=-h,g=1+h,v=[_,_,g,_,g,g,_,_,g,g,_,g],b=6,R=6,P=3,E=2,m=1,U=new Float32Array(P*R*b),L=new Float32Array(E*R*b),N=new Float32Array(m*R*b);for(let A=0;A<b;A++){const V=A%3*2/3-1,Z=A>2?0:-1,D=[V,Z,0,V+2/3,Z,0,V+2/3,Z+1,0,V,Z,0,V+2/3,Z+1,0,V,Z+1,0];U.set(D,P*R*A),L.set(v,E*R*A);const y=[A,A,A,A,A,A];N.set(y,m*R*A)}const z=new fn;z.setAttribute("position",new An(U,P)),z.setAttribute("uv",new An(L,E)),z.setAttribute("faceIndex",new An(N,m)),n.push(new li(z,null)),a>Si&&a--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function ql(r,e,t){const n=new $i(r,e,t);return n.texture.mapping=ua,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function ur(r,e,t,n,a){r.viewport.set(e,t,n,a),r.scissor.set(e,t,n,a)}function Rv(r,e,t){return new ui({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:Tv,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${r}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:da(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 3.2: Transform view direction to hemisphere configuration
				vec3 Vh = normalize(vec3(alpha * V.x, alpha * V.y, V.z));

				// Section 4.1: Orthonormal basis
				float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
				vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(Vh, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + Vh.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:ai,depthTest:!1,depthWrite:!1})}function Cv(r,e,t){const n=new Float32Array(Gi),a=new K(0,1,0);return new ui({name:"SphericalGaussianBlur",defines:{n:Gi,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${r}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:a}},vertexShader:da(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:ai,depthTest:!1,depthWrite:!1})}function Kl(){return new ui({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:da(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:ai,depthTest:!1,depthWrite:!1})}function Zl(){return new ui({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:da(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:ai,depthTest:!1,depthWrite:!1})}function da(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function Pv(r){let e=new WeakMap,t=null;function n(f){if(f&&f.isTexture){const p=f.mapping,h=p===Eo||p===Mo,_=p===Er||p===Mr;if(h||_){let g=e.get(f);const v=g!==void 0?g.texture.pmremVersion:0;if(f.isRenderTargetTexture&&f.pmremVersion!==v)return t===null&&(t=new Yl(r)),g=h?t.fromEquirectangular(f,g):t.fromCubemap(f,g),g.texture.pmremVersion=f.pmremVersion,e.set(f,g),g.texture;if(g!==void 0)return g.texture;{const b=f.image;return h&&b&&b.height>0||_&&b&&a(b)?(t===null&&(t=new Yl(r)),g=h?t.fromEquirectangular(f):t.fromCubemap(f),g.texture.pmremVersion=f.pmremVersion,e.set(f,g),f.addEventListener("dispose",o),g.texture):null}}}return f}function a(f){let p=0;const h=6;for(let _=0;_<h;_++)f[_]!==void 0&&p++;return p===h}function o(f){const p=f.target;p.removeEventListener("dispose",o);const h=e.get(p);h!==void 0&&(e.delete(p),h.dispose())}function l(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:l}}function Dv(r){const e={};function t(n){if(e[n]!==void 0)return e[n];const a=r.getExtension(n);return e[n]=a,a}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const a=t(n);return a===null&&Zr("WebGLRenderer: "+n+" extension not supported."),a}}}function Lv(r,e,t,n){const a={},o=new WeakMap;function l(g){const v=g.target;v.index!==null&&e.remove(v.index);for(const R in v.attributes)e.remove(v.attributes[R]);v.removeEventListener("dispose",l),delete a[v.id];const b=o.get(v);b&&(e.remove(b),o.delete(v)),n.releaseStatesOfGeometry(v),v.isInstancedBufferGeometry===!0&&delete v._maxInstanceCount,t.memory.geometries--}function f(g,v){return a[v.id]===!0||(v.addEventListener("dispose",l),a[v.id]=!0,t.memory.geometries++),v}function p(g){const v=g.attributes;for(const b in v)e.update(v[b],r.ARRAY_BUFFER)}function h(g){const v=[],b=g.index,R=g.attributes.position;let P=0;if(b!==null){const U=b.array;P=b.version;for(let L=0,N=U.length;L<N;L+=3){const z=U[L+0],A=U[L+1],V=U[L+2];v.push(z,A,A,V,V,z)}}else if(R!==void 0){const U=R.array;P=R.version;for(let L=0,N=U.length/3-1;L<N;L+=3){const z=L+0,A=L+1,V=L+2;v.push(z,A,A,V,V,z)}}else return;const E=new(Fu(v)?Bu:ku)(v,1);E.version=P;const m=o.get(g);m&&e.remove(m),o.set(g,E)}function _(g){const v=o.get(g);if(v){const b=g.index;b!==null&&v.version<b.version&&h(g)}else h(g);return o.get(g)}return{get:f,update:p,getWireframeAttribute:_}}function Fv(r,e,t){let n;function a(v){n=v}let o,l;function f(v){o=v.type,l=v.bytesPerElement}function p(v,b){r.drawElements(n,b,o,v*l),t.update(b,n,1)}function h(v,b,R){R!==0&&(r.drawElementsInstanced(n,b,o,v*l,R),t.update(b,n,R))}function _(v,b,R){if(R===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,b,0,o,v,0,R);let E=0;for(let m=0;m<R;m++)E+=b[m];t.update(E,n,1)}function g(v,b,R,P){if(R===0)return;const E=e.get("WEBGL_multi_draw");if(E===null)for(let m=0;m<v.length;m++)h(v[m]/l,b[m],P[m]);else{E.multiDrawElementsInstancedWEBGL(n,b,0,o,v,0,P,0,R);let m=0;for(let U=0;U<R;U++)m+=b[U]*P[U];t.update(m,n,1)}}this.setMode=a,this.setIndex=f,this.render=p,this.renderInstances=h,this.renderMultiDraw=_,this.renderMultiDrawInstances=g}function Iv(r){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(o,l,f){switch(t.calls++,l){case r.TRIANGLES:t.triangles+=f*(o/3);break;case r.LINES:t.lines+=f*(o/2);break;case r.LINE_STRIP:t.lines+=f*(o-1);break;case r.LINE_LOOP:t.lines+=f*o;break;case r.POINTS:t.points+=f*o;break;default:Gt("WebGLInfo: Unknown draw mode:",l);break}}function a(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:a,update:n}}function Uv(r,e,t){const n=new WeakMap,a=new zt;function o(l,f,p){const h=l.morphTargetInfluences,_=f.morphAttributes.position||f.morphAttributes.normal||f.morphAttributes.color,g=_!==void 0?_.length:0;let v=n.get(f);if(v===void 0||v.count!==g){let D=function(){V.dispose(),n.delete(f),f.removeEventListener("dispose",D)};v!==void 0&&v.texture.dispose();const b=f.morphAttributes.position!==void 0,R=f.morphAttributes.normal!==void 0,P=f.morphAttributes.color!==void 0,E=f.morphAttributes.position||[],m=f.morphAttributes.normal||[],U=f.morphAttributes.color||[];let L=0;b===!0&&(L=1),R===!0&&(L=2),P===!0&&(L=3);let N=f.attributes.position.count*L,z=1;N>e.maxTextureSize&&(z=Math.ceil(N/e.maxTextureSize),N=e.maxTextureSize);const A=new Float32Array(N*z*4*g),V=new Iu(A,N,z,g);V.type=ii,V.needsUpdate=!0;const Z=L*4;for(let y=0;y<g;y++){const G=E[y],Y=m[y],J=U[y],ee=N*z*4*y;for(let fe=0;fe<G.count;fe++){const le=fe*Z;b===!0&&(a.fromBufferAttribute(G,fe),A[ee+le+0]=a.x,A[ee+le+1]=a.y,A[ee+le+2]=a.z,A[ee+le+3]=0),R===!0&&(a.fromBufferAttribute(Y,fe),A[ee+le+4]=a.x,A[ee+le+5]=a.y,A[ee+le+6]=a.z,A[ee+le+7]=0),P===!0&&(a.fromBufferAttribute(J,fe),A[ee+le+8]=a.x,A[ee+le+9]=a.y,A[ee+le+10]=a.z,A[ee+le+11]=J.itemSize===4?a.w:1)}}v={count:g,texture:V,size:new et(N,z)},n.set(f,v),f.addEventListener("dispose",D)}if(l.isInstancedMesh===!0&&l.morphTexture!==null)p.getUniforms().setValue(r,"morphTexture",l.morphTexture,t);else{let b=0;for(let P=0;P<h.length;P++)b+=h[P];const R=f.morphTargetsRelative?1:1-b;p.getUniforms().setValue(r,"morphTargetBaseInfluence",R),p.getUniforms().setValue(r,"morphTargetInfluences",h)}p.getUniforms().setValue(r,"morphTargetsTexture",v.texture,t),p.getUniforms().setValue(r,"morphTargetsTextureSize",v.size)}return{update:o}}function Nv(r,e,t,n){let a=new WeakMap;function o(p){const h=n.render.frame,_=p.geometry,g=e.get(p,_);if(a.get(g)!==h&&(e.update(g),a.set(g,h)),p.isInstancedMesh&&(p.hasEventListener("dispose",f)===!1&&p.addEventListener("dispose",f),a.get(p)!==h&&(t.update(p.instanceMatrix,r.ARRAY_BUFFER),p.instanceColor!==null&&t.update(p.instanceColor,r.ARRAY_BUFFER),a.set(p,h))),p.isSkinnedMesh){const v=p.skeleton;a.get(v)!==h&&(v.update(),a.set(v,h))}return g}function l(){a=new WeakMap}function f(p){const h=p.target;h.removeEventListener("dispose",f),t.remove(h.instanceMatrix),h.instanceColor!==null&&t.remove(h.instanceColor)}return{update:o,dispose:l}}const qu=new un,Jl=new Wu(1,1),Ku=new Iu,Zu=new i_,Ju=new Gu,Ql=[],eu=[],tu=new Float32Array(16),nu=new Float32Array(9),iu=new Float32Array(4);function Dr(r,e,t){const n=r[0];if(n<=0||n>0)return r;const a=e*t;let o=Ql[a];if(o===void 0&&(o=new Float32Array(a),Ql[a]=o),e!==0){n.toArray(o,0);for(let l=1,f=0;l!==e;++l)f+=t,r[l].toArray(o,f)}return o}function Yt(r,e){if(r.length!==e.length)return!1;for(let t=0,n=r.length;t<n;t++)if(r[t]!==e[t])return!1;return!0}function qt(r,e){for(let t=0,n=e.length;t<n;t++)r[t]=e[t]}function ha(r,e){let t=eu[e];t===void 0&&(t=new Int32Array(e),eu[e]=t);for(let n=0;n!==e;++n)t[n]=r.allocateTextureUnit();return t}function Ov(r,e){const t=this.cache;t[0]!==e&&(r.uniform1f(this.addr,e),t[0]=e)}function kv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(r.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Yt(t,e))return;r.uniform2fv(this.addr,e),qt(t,e)}}function Bv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(r.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(r.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Yt(t,e))return;r.uniform3fv(this.addr,e),qt(t,e)}}function zv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(r.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Yt(t,e))return;r.uniform4fv(this.addr,e),qt(t,e)}}function Vv(r,e){const t=this.cache,n=e.elements;if(n===void 0){if(Yt(t,e))return;r.uniformMatrix2fv(this.addr,!1,e),qt(t,e)}else{if(Yt(t,n))return;iu.set(n),r.uniformMatrix2fv(this.addr,!1,iu),qt(t,n)}}function Gv(r,e){const t=this.cache,n=e.elements;if(n===void 0){if(Yt(t,e))return;r.uniformMatrix3fv(this.addr,!1,e),qt(t,e)}else{if(Yt(t,n))return;nu.set(n),r.uniformMatrix3fv(this.addr,!1,nu),qt(t,n)}}function Hv(r,e){const t=this.cache,n=e.elements;if(n===void 0){if(Yt(t,e))return;r.uniformMatrix4fv(this.addr,!1,e),qt(t,e)}else{if(Yt(t,n))return;tu.set(n),r.uniformMatrix4fv(this.addr,!1,tu),qt(t,n)}}function Wv(r,e){const t=this.cache;t[0]!==e&&(r.uniform1i(this.addr,e),t[0]=e)}function Xv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(r.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Yt(t,e))return;r.uniform2iv(this.addr,e),qt(t,e)}}function $v(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(r.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Yt(t,e))return;r.uniform3iv(this.addr,e),qt(t,e)}}function jv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(r.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Yt(t,e))return;r.uniform4iv(this.addr,e),qt(t,e)}}function Yv(r,e){const t=this.cache;t[0]!==e&&(r.uniform1ui(this.addr,e),t[0]=e)}function qv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(r.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Yt(t,e))return;r.uniform2uiv(this.addr,e),qt(t,e)}}function Kv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(r.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Yt(t,e))return;r.uniform3uiv(this.addr,e),qt(t,e)}}function Zv(r,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(r.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Yt(t,e))return;r.uniform4uiv(this.addr,e),qt(t,e)}}function Jv(r,e,t){const n=this.cache,a=t.allocateTextureUnit();n[0]!==a&&(r.uniform1i(this.addr,a),n[0]=a);let o;this.type===r.SAMPLER_2D_SHADOW?(Jl.compareFunction=Lu,o=Jl):o=qu,t.setTexture2D(e||o,a)}function Qv(r,e,t){const n=this.cache,a=t.allocateTextureUnit();n[0]!==a&&(r.uniform1i(this.addr,a),n[0]=a),t.setTexture3D(e||Zu,a)}function ey(r,e,t){const n=this.cache,a=t.allocateTextureUnit();n[0]!==a&&(r.uniform1i(this.addr,a),n[0]=a),t.setTextureCube(e||Ju,a)}function ty(r,e,t){const n=this.cache,a=t.allocateTextureUnit();n[0]!==a&&(r.uniform1i(this.addr,a),n[0]=a),t.setTexture2DArray(e||Ku,a)}function ny(r){switch(r){case 5126:return Ov;case 35664:return kv;case 35665:return Bv;case 35666:return zv;case 35674:return Vv;case 35675:return Gv;case 35676:return Hv;case 5124:case 35670:return Wv;case 35667:case 35671:return Xv;case 35668:case 35672:return $v;case 35669:case 35673:return jv;case 5125:return Yv;case 36294:return qv;case 36295:return Kv;case 36296:return Zv;case 35678:case 36198:case 36298:case 36306:case 35682:return Jv;case 35679:case 36299:case 36307:return Qv;case 35680:case 36300:case 36308:case 36293:return ey;case 36289:case 36303:case 36311:case 36292:return ty}}function iy(r,e){r.uniform1fv(this.addr,e)}function ry(r,e){const t=Dr(e,this.size,2);r.uniform2fv(this.addr,t)}function sy(r,e){const t=Dr(e,this.size,3);r.uniform3fv(this.addr,t)}function ay(r,e){const t=Dr(e,this.size,4);r.uniform4fv(this.addr,t)}function oy(r,e){const t=Dr(e,this.size,4);r.uniformMatrix2fv(this.addr,!1,t)}function cy(r,e){const t=Dr(e,this.size,9);r.uniformMatrix3fv(this.addr,!1,t)}function ly(r,e){const t=Dr(e,this.size,16);r.uniformMatrix4fv(this.addr,!1,t)}function uy(r,e){r.uniform1iv(this.addr,e)}function fy(r,e){r.uniform2iv(this.addr,e)}function dy(r,e){r.uniform3iv(this.addr,e)}function hy(r,e){r.uniform4iv(this.addr,e)}function py(r,e){r.uniform1uiv(this.addr,e)}function my(r,e){r.uniform2uiv(this.addr,e)}function _y(r,e){r.uniform3uiv(this.addr,e)}function xy(r,e){r.uniform4uiv(this.addr,e)}function gy(r,e,t){const n=this.cache,a=e.length,o=ha(t,a);Yt(n,o)||(r.uniform1iv(this.addr,o),qt(n,o));for(let l=0;l!==a;++l)t.setTexture2D(e[l]||qu,o[l])}function vy(r,e,t){const n=this.cache,a=e.length,o=ha(t,a);Yt(n,o)||(r.uniform1iv(this.addr,o),qt(n,o));for(let l=0;l!==a;++l)t.setTexture3D(e[l]||Zu,o[l])}function yy(r,e,t){const n=this.cache,a=e.length,o=ha(t,a);Yt(n,o)||(r.uniform1iv(this.addr,o),qt(n,o));for(let l=0;l!==a;++l)t.setTextureCube(e[l]||Ju,o[l])}function Sy(r,e,t){const n=this.cache,a=e.length,o=ha(t,a);Yt(n,o)||(r.uniform1iv(this.addr,o),qt(n,o));for(let l=0;l!==a;++l)t.setTexture2DArray(e[l]||Ku,o[l])}function by(r){switch(r){case 5126:return iy;case 35664:return ry;case 35665:return sy;case 35666:return ay;case 35674:return oy;case 35675:return cy;case 35676:return ly;case 5124:case 35670:return uy;case 35667:case 35671:return fy;case 35668:case 35672:return dy;case 35669:case 35673:return hy;case 5125:return py;case 36294:return my;case 36295:return _y;case 36296:return xy;case 35678:case 36198:case 36298:case 36306:case 35682:return gy;case 35679:case 36299:case 36307:return vy;case 35680:case 36300:case 36308:case 36293:return yy;case 36289:case 36303:case 36311:case 36292:return Sy}}class Ey{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=ny(t.type)}}class My{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=by(t.type)}}class Ty{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const a=this.seq;for(let o=0,l=a.length;o!==l;++o){const f=a[o];f.setValue(e,t[f.id],n)}}}const uo=/(\w+)(\])?(\[|\.)?/g;function ru(r,e){r.seq.push(e),r.map[e.id]=e}function wy(r,e,t){const n=r.name,a=n.length;for(uo.lastIndex=0;;){const o=uo.exec(n),l=uo.lastIndex;let f=o[1];const p=o[2]==="]",h=o[3];if(p&&(f=f|0),h===void 0||h==="["&&l+2===a){ru(t,h===void 0?new Ey(f,r,e):new My(f,r,e));break}else{let g=t.map[f];g===void 0&&(g=new Ty(f),ru(t,g)),t=g}}}class ta{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let a=0;a<n;++a){const o=e.getActiveUniform(t,a),l=e.getUniformLocation(t,o.name);wy(o,l,this)}}setValue(e,t,n,a){const o=this.map[t];o!==void 0&&o.setValue(e,n,a)}setOptional(e,t,n){const a=t[n];a!==void 0&&this.setValue(e,n,a)}static upload(e,t,n,a){for(let o=0,l=t.length;o!==l;++o){const f=t[o],p=n[f.id];p.needsUpdate!==!1&&f.setValue(e,p.value,a)}}static seqWithValue(e,t){const n=[];for(let a=0,o=e.length;a!==o;++a){const l=e[a];l.id in t&&n.push(l)}return n}}function su(r,e,t){const n=r.createShader(e);return r.shaderSource(n,t),r.compileShader(n),n}const Ay=37297;let Ry=0;function Cy(r,e){const t=r.split(`
`),n=[],a=Math.max(e-6,0),o=Math.min(e+6,t.length);for(let l=a;l<o;l++){const f=l+1;n.push(`${f===e?">":" "} ${f}: ${t[l]}`)}return n.join(`
`)}const au=new tt;function Py(r){yt._getMatrix(au,yt.workingColorSpace,r);const e=`mat3( ${au.elements.map(t=>t.toFixed(4))} )`;switch(yt.getTransfer(r)){case ia:return[e,"LinearTransferOETF"];case Ft:return[e,"sRGBTransferOETF"];default:return Ze("WebGLProgram: Unsupported color space: ",r),[e,"LinearTransferOETF"]}}function ou(r,e,t){const n=r.getShaderParameter(e,r.COMPILE_STATUS),o=(r.getShaderInfoLog(e)||"").trim();if(n&&o==="")return"";const l=/ERROR: 0:(\d+)/.exec(o);if(l){const f=parseInt(l[1]);return t.toUpperCase()+`

`+o+`

`+Cy(r.getShaderSource(e),f)}else return o}function Dy(r,e){const t=Py(e);return[`vec4 ${r}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function Ly(r,e){let t;switch(e){case C0:t="Linear";break;case P0:t="Reinhard";break;case D0:t="Cineon";break;case L0:t="ACESFilmic";break;case I0:t="AgX";break;case U0:t="Neutral";break;case F0:t="Custom";break;default:Ze("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+r+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const js=new K;function Fy(){yt.getLuminanceCoefficients(js);const r=js.x.toFixed(4),e=js.y.toFixed(4),t=js.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${r}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function Iy(r){return[r.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",r.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Xr).join(`
`)}function Uy(r){const e=[];for(const t in r){const n=r[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function Ny(r,e){const t={},n=r.getProgramParameter(e,r.ACTIVE_ATTRIBUTES);for(let a=0;a<n;a++){const o=r.getActiveAttrib(e,a),l=o.name;let f=1;o.type===r.FLOAT_MAT2&&(f=2),o.type===r.FLOAT_MAT3&&(f=3),o.type===r.FLOAT_MAT4&&(f=4),t[l]={type:o.type,location:r.getAttribLocation(e,l),locationSize:f}}return t}function Xr(r){return r!==""}function cu(r,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return r.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function lu(r,e){return r.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const Oy=/^[ \t]*#include +<([\w\d./]+)>/gm;function nc(r){return r.replace(Oy,By)}const ky=new Map;function By(r,e){let t=st[e];if(t===void 0){const n=ky.get(e);if(n!==void 0)t=st[n],Ze('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return nc(t)}const zy=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function uu(r){return r.replace(zy,Vy)}function Vy(r,e,t,n){let a="";for(let o=parseInt(e);o<parseInt(t);o++)a+=n.replace(/\[\s*i\s*\]/g,"[ "+o+" ]").replace(/UNROLLED_LOOP_INDEX/g,o);return a}function fu(r){let e=`precision ${r.precision} float;
	precision ${r.precision} int;
	precision ${r.precision} sampler2D;
	precision ${r.precision} samplerCube;
	precision ${r.precision} sampler3D;
	precision ${r.precision} sampler2DArray;
	precision ${r.precision} sampler2DShadow;
	precision ${r.precision} samplerCubeShadow;
	precision ${r.precision} sampler2DArrayShadow;
	precision ${r.precision} isampler2D;
	precision ${r.precision} isampler3D;
	precision ${r.precision} isamplerCube;
	precision ${r.precision} isampler2DArray;
	precision ${r.precision} usampler2D;
	precision ${r.precision} usampler3D;
	precision ${r.precision} usamplerCube;
	precision ${r.precision} usampler2DArray;
	`;return r.precision==="highp"?e+=`
#define HIGH_PRECISION`:r.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:r.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}function Gy(r){let e="SHADOWMAP_TYPE_BASIC";return r.shadowMapType===Su?e="SHADOWMAP_TYPE_PCF":r.shadowMapType===c0?e="SHADOWMAP_TYPE_PCF_SOFT":r.shadowMapType===ti&&(e="SHADOWMAP_TYPE_VSM"),e}function Hy(r){let e="ENVMAP_TYPE_CUBE";if(r.envMap)switch(r.envMapMode){case Er:case Mr:e="ENVMAP_TYPE_CUBE";break;case ua:e="ENVMAP_TYPE_CUBE_UV";break}return e}function Wy(r){let e="ENVMAP_MODE_REFLECTION";if(r.envMap)switch(r.envMapMode){case Mr:e="ENVMAP_MODE_REFRACTION";break}return e}function Xy(r){let e="ENVMAP_BLENDING_NONE";if(r.envMap)switch(r.combine){case bu:e="ENVMAP_BLENDING_MULTIPLY";break;case A0:e="ENVMAP_BLENDING_MIX";break;case R0:e="ENVMAP_BLENDING_ADD";break}return e}function $y(r){const e=r.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function jy(r,e,t,n){const a=r.getContext(),o=t.defines;let l=t.vertexShader,f=t.fragmentShader;const p=Gy(t),h=Hy(t),_=Wy(t),g=Xy(t),v=$y(t),b=Iy(t),R=Uy(o),P=a.createProgram();let E,m,U=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(E=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,R].filter(Xr).join(`
`),E.length>0&&(E+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,R].filter(Xr).join(`
`),m.length>0&&(m+=`
`)):(E=[fu(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,R,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+_:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+p:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Xr).join(`
`),m=[fu(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,R,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+h:"",t.envMap?"#define "+_:"",t.envMap?"#define "+g:"",v?"#define CUBEUV_TEXEL_WIDTH "+v.texelWidth:"",v?"#define CUBEUV_TEXEL_HEIGHT "+v.texelHeight:"",v?"#define CUBEUV_MAX_MIP "+v.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+p:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==bi?"#define TONE_MAPPING":"",t.toneMapping!==bi?st.tonemapping_pars_fragment:"",t.toneMapping!==bi?Ly("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",st.colorspace_pars_fragment,Dy("linearToOutputTexel",t.outputColorSpace),Fy(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Xr).join(`
`)),l=nc(l),l=cu(l,t),l=lu(l,t),f=nc(f),f=cu(f,t),f=lu(f,t),l=uu(l),f=uu(f),t.isRawShaderMaterial!==!0&&(U=`#version 300 es
`,E=[b,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+E,m=["#define varying in",t.glslVersion===vl?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===vl?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const L=U+E+l,N=U+m+f,z=su(a,a.VERTEX_SHADER,L),A=su(a,a.FRAGMENT_SHADER,N);a.attachShader(P,z),a.attachShader(P,A),t.index0AttributeName!==void 0?a.bindAttribLocation(P,0,t.index0AttributeName):t.morphTargets===!0&&a.bindAttribLocation(P,0,"position"),a.linkProgram(P);function V(G){if(r.debug.checkShaderErrors){const Y=a.getProgramInfoLog(P)||"",J=a.getShaderInfoLog(z)||"",ee=a.getShaderInfoLog(A)||"",fe=Y.trim(),le=J.trim(),Ee=ee.trim();let se=!0,Me=!0;if(a.getProgramParameter(P,a.LINK_STATUS)===!1)if(se=!1,typeof r.debug.onShaderError=="function")r.debug.onShaderError(a,P,z,A);else{const we=ou(a,z,"vertex"),Ge=ou(a,A,"fragment");Gt("THREE.WebGLProgram: Shader Error "+a.getError()+" - VALIDATE_STATUS "+a.getProgramParameter(P,a.VALIDATE_STATUS)+`

Material Name: `+G.name+`
Material Type: `+G.type+`

Program Info Log: `+fe+`
`+we+`
`+Ge)}else fe!==""?Ze("WebGLProgram: Program Info Log:",fe):(le===""||Ee==="")&&(Me=!1);Me&&(G.diagnostics={runnable:se,programLog:fe,vertexShader:{log:le,prefix:E},fragmentShader:{log:Ee,prefix:m}})}a.deleteShader(z),a.deleteShader(A),Z=new ta(a,P),D=Ny(a,P)}let Z;this.getUniforms=function(){return Z===void 0&&V(this),Z};let D;this.getAttributes=function(){return D===void 0&&V(this),D};let y=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return y===!1&&(y=a.getProgramParameter(P,Ay)),y},this.destroy=function(){n.releaseStatesOfProgram(this),a.deleteProgram(P),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=Ry++,this.cacheKey=e,this.usedTimes=1,this.program=P,this.vertexShader=z,this.fragmentShader=A,this}let Yy=0;class qy{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,a=this._getShaderStage(t),o=this._getShaderStage(n),l=this._getShaderCacheForMaterial(e);return l.has(a)===!1&&(l.add(a),a.usedTimes++),l.has(o)===!1&&(l.add(o),o.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new Ky(e),t.set(e,n)),n}}class Ky{constructor(e){this.id=Yy++,this.code=e,this.usedTimes=0}}function Zy(r,e,t,n,a,o,l){const f=new Uu,p=new qy,h=new Set,_=[],g=a.logarithmicDepthBuffer,v=a.vertexTextures;let b=a.precision;const R={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function P(D){return h.add(D),D===0?"uv":`uv${D}`}function E(D,y,G,Y,J){const ee=Y.fog,fe=J.geometry,le=D.isMeshStandardMaterial?Y.environment:null,Ee=(D.isMeshStandardMaterial?t:e).get(D.envMap||le),se=Ee&&Ee.mapping===ua?Ee.image.height:null,Me=R[D.type];D.precision!==null&&(b=a.getMaxPrecision(D.precision),b!==D.precision&&Ze("WebGLProgram.getParameters:",D.precision,"not supported, using",b,"instead."));const we=fe.morphAttributes.position||fe.morphAttributes.normal||fe.morphAttributes.color,Ge=we!==void 0?we.length:0;let ot=0;fe.morphAttributes.position!==void 0&&(ot=1),fe.morphAttributes.normal!==void 0&&(ot=2),fe.morphAttributes.color!==void 0&&(ot=3);let _t,Je,at,ce;if(Me){const ut=Gn[Me];_t=ut.vertexShader,Je=ut.fragmentShader}else _t=D.vertexShader,Je=D.fragmentShader,p.update(D),at=p.getVertexShaderID(D),ce=p.getFragmentShaderID(D);const de=r.getRenderTarget(),ie=r.state.buffers.depth.getReversed(),ye=J.isInstancedMesh===!0,Be=J.isBatchedMesh===!0,nt=!!D.map,Pt=!!D.matcap,it=!!Ee,Mt=!!D.aoMap,O=!!D.lightMap,ct=!!D.bumpMap,lt=!!D.normalMap,Dt=!!D.displacementMap,Ie=!!D.emissiveMap,Rt=!!D.metalnessMap,Ne=!!D.roughnessMap,$e=D.anisotropy>0,F=D.clearcoat>0,w=D.dispersion>0,$=D.iridescence>0,re=D.sheen>0,he=D.transmission>0,ne=$e&&!!D.anisotropyMap,ke=F&&!!D.clearcoatMap,Ae=F&&!!D.clearcoatNormalMap,He=F&&!!D.clearcoatRoughnessMap,Oe=$&&!!D.iridescenceMap,me=$&&!!D.iridescenceThicknessMap,be=re&&!!D.sheenColorMap,je=re&&!!D.sheenRoughnessMap,We=!!D.specularMap,De=!!D.specularColorMap,qe=!!D.specularIntensityMap,k=he&&!!D.transmissionMap,Re=he&&!!D.thicknessMap,Te=!!D.gradientMap,Se=!!D.alphaMap,H=D.alphaTest>0,B=!!D.alphaHash,Le=!!D.extensions;let Ke=bi;D.toneMapped&&(de===null||de.isXRRenderTarget===!0)&&(Ke=r.toneMapping);const vt={shaderID:Me,shaderType:D.type,shaderName:D.name,vertexShader:_t,fragmentShader:Je,defines:D.defines,customVertexShaderID:at,customFragmentShaderID:ce,isRawShaderMaterial:D.isRawShaderMaterial===!0,glslVersion:D.glslVersion,precision:b,batching:Be,batchingColor:Be&&J._colorsTexture!==null,instancing:ye,instancingColor:ye&&J.instanceColor!==null,instancingMorph:ye&&J.morphTexture!==null,supportsVertexTextures:v,outputColorSpace:de===null?r.outputColorSpace:de.isXRRenderTarget===!0?de.texture.colorSpace:Tr,alphaToCoverage:!!D.alphaToCoverage,map:nt,matcap:Pt,envMap:it,envMapMode:it&&Ee.mapping,envMapCubeUVHeight:se,aoMap:Mt,lightMap:O,bumpMap:ct,normalMap:lt,displacementMap:v&&Dt,emissiveMap:Ie,normalMapObjectSpace:lt&&D.normalMapType===B0,normalMapTangentSpace:lt&&D.normalMapType===Du,metalnessMap:Rt,roughnessMap:Ne,anisotropy:$e,anisotropyMap:ne,clearcoat:F,clearcoatMap:ke,clearcoatNormalMap:Ae,clearcoatRoughnessMap:He,dispersion:w,iridescence:$,iridescenceMap:Oe,iridescenceThicknessMap:me,sheen:re,sheenColorMap:be,sheenRoughnessMap:je,specularMap:We,specularColorMap:De,specularIntensityMap:qe,transmission:he,transmissionMap:k,thicknessMap:Re,gradientMap:Te,opaque:D.transparent===!1&&D.blending===gr&&D.alphaToCoverage===!1,alphaMap:Se,alphaTest:H,alphaHash:B,combine:D.combine,mapUv:nt&&P(D.map.channel),aoMapUv:Mt&&P(D.aoMap.channel),lightMapUv:O&&P(D.lightMap.channel),bumpMapUv:ct&&P(D.bumpMap.channel),normalMapUv:lt&&P(D.normalMap.channel),displacementMapUv:Dt&&P(D.displacementMap.channel),emissiveMapUv:Ie&&P(D.emissiveMap.channel),metalnessMapUv:Rt&&P(D.metalnessMap.channel),roughnessMapUv:Ne&&P(D.roughnessMap.channel),anisotropyMapUv:ne&&P(D.anisotropyMap.channel),clearcoatMapUv:ke&&P(D.clearcoatMap.channel),clearcoatNormalMapUv:Ae&&P(D.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:He&&P(D.clearcoatRoughnessMap.channel),iridescenceMapUv:Oe&&P(D.iridescenceMap.channel),iridescenceThicknessMapUv:me&&P(D.iridescenceThicknessMap.channel),sheenColorMapUv:be&&P(D.sheenColorMap.channel),sheenRoughnessMapUv:je&&P(D.sheenRoughnessMap.channel),specularMapUv:We&&P(D.specularMap.channel),specularColorMapUv:De&&P(D.specularColorMap.channel),specularIntensityMapUv:qe&&P(D.specularIntensityMap.channel),transmissionMapUv:k&&P(D.transmissionMap.channel),thicknessMapUv:Re&&P(D.thicknessMap.channel),alphaMapUv:Se&&P(D.alphaMap.channel),vertexTangents:!!fe.attributes.tangent&&(lt||$e),vertexColors:D.vertexColors,vertexAlphas:D.vertexColors===!0&&!!fe.attributes.color&&fe.attributes.color.itemSize===4,pointsUvs:J.isPoints===!0&&!!fe.attributes.uv&&(nt||Se),fog:!!ee,useFog:D.fog===!0,fogExp2:!!ee&&ee.isFogExp2,flatShading:D.flatShading===!0&&D.wireframe===!1,sizeAttenuation:D.sizeAttenuation===!0,logarithmicDepthBuffer:g,reversedDepthBuffer:ie,skinning:J.isSkinnedMesh===!0,morphTargets:fe.morphAttributes.position!==void 0,morphNormals:fe.morphAttributes.normal!==void 0,morphColors:fe.morphAttributes.color!==void 0,morphTargetsCount:Ge,morphTextureStride:ot,numDirLights:y.directional.length,numPointLights:y.point.length,numSpotLights:y.spot.length,numSpotLightMaps:y.spotLightMap.length,numRectAreaLights:y.rectArea.length,numHemiLights:y.hemi.length,numDirLightShadows:y.directionalShadowMap.length,numPointLightShadows:y.pointShadowMap.length,numSpotLightShadows:y.spotShadowMap.length,numSpotLightShadowsWithMaps:y.numSpotLightShadowsWithMaps,numLightProbes:y.numLightProbes,numClippingPlanes:l.numPlanes,numClipIntersection:l.numIntersection,dithering:D.dithering,shadowMapEnabled:r.shadowMap.enabled&&G.length>0,shadowMapType:r.shadowMap.type,toneMapping:Ke,decodeVideoTexture:nt&&D.map.isVideoTexture===!0&&yt.getTransfer(D.map.colorSpace)===Ft,decodeVideoTextureEmissive:Ie&&D.emissiveMap.isVideoTexture===!0&&yt.getTransfer(D.emissiveMap.colorSpace)===Ft,premultipliedAlpha:D.premultipliedAlpha,doubleSided:D.side===Hn,flipSided:D.side===_n,useDepthPacking:D.depthPacking>=0,depthPacking:D.depthPacking||0,index0AttributeName:D.index0AttributeName,extensionClipCullDistance:Le&&D.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Le&&D.extensions.multiDraw===!0||Be)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:D.customProgramCacheKey()};return vt.vertexUv1s=h.has(1),vt.vertexUv2s=h.has(2),vt.vertexUv3s=h.has(3),h.clear(),vt}function m(D){const y=[];if(D.shaderID?y.push(D.shaderID):(y.push(D.customVertexShaderID),y.push(D.customFragmentShaderID)),D.defines!==void 0)for(const G in D.defines)y.push(G),y.push(D.defines[G]);return D.isRawShaderMaterial===!1&&(U(y,D),L(y,D),y.push(r.outputColorSpace)),y.push(D.customProgramCacheKey),y.join()}function U(D,y){D.push(y.precision),D.push(y.outputColorSpace),D.push(y.envMapMode),D.push(y.envMapCubeUVHeight),D.push(y.mapUv),D.push(y.alphaMapUv),D.push(y.lightMapUv),D.push(y.aoMapUv),D.push(y.bumpMapUv),D.push(y.normalMapUv),D.push(y.displacementMapUv),D.push(y.emissiveMapUv),D.push(y.metalnessMapUv),D.push(y.roughnessMapUv),D.push(y.anisotropyMapUv),D.push(y.clearcoatMapUv),D.push(y.clearcoatNormalMapUv),D.push(y.clearcoatRoughnessMapUv),D.push(y.iridescenceMapUv),D.push(y.iridescenceThicknessMapUv),D.push(y.sheenColorMapUv),D.push(y.sheenRoughnessMapUv),D.push(y.specularMapUv),D.push(y.specularColorMapUv),D.push(y.specularIntensityMapUv),D.push(y.transmissionMapUv),D.push(y.thicknessMapUv),D.push(y.combine),D.push(y.fogExp2),D.push(y.sizeAttenuation),D.push(y.morphTargetsCount),D.push(y.morphAttributeCount),D.push(y.numDirLights),D.push(y.numPointLights),D.push(y.numSpotLights),D.push(y.numSpotLightMaps),D.push(y.numHemiLights),D.push(y.numRectAreaLights),D.push(y.numDirLightShadows),D.push(y.numPointLightShadows),D.push(y.numSpotLightShadows),D.push(y.numSpotLightShadowsWithMaps),D.push(y.numLightProbes),D.push(y.shadowMapType),D.push(y.toneMapping),D.push(y.numClippingPlanes),D.push(y.numClipIntersection),D.push(y.depthPacking)}function L(D,y){f.disableAll(),y.supportsVertexTextures&&f.enable(0),y.instancing&&f.enable(1),y.instancingColor&&f.enable(2),y.instancingMorph&&f.enable(3),y.matcap&&f.enable(4),y.envMap&&f.enable(5),y.normalMapObjectSpace&&f.enable(6),y.normalMapTangentSpace&&f.enable(7),y.clearcoat&&f.enable(8),y.iridescence&&f.enable(9),y.alphaTest&&f.enable(10),y.vertexColors&&f.enable(11),y.vertexAlphas&&f.enable(12),y.vertexUv1s&&f.enable(13),y.vertexUv2s&&f.enable(14),y.vertexUv3s&&f.enable(15),y.vertexTangents&&f.enable(16),y.anisotropy&&f.enable(17),y.alphaHash&&f.enable(18),y.batching&&f.enable(19),y.dispersion&&f.enable(20),y.batchingColor&&f.enable(21),y.gradientMap&&f.enable(22),D.push(f.mask),f.disableAll(),y.fog&&f.enable(0),y.useFog&&f.enable(1),y.flatShading&&f.enable(2),y.logarithmicDepthBuffer&&f.enable(3),y.reversedDepthBuffer&&f.enable(4),y.skinning&&f.enable(5),y.morphTargets&&f.enable(6),y.morphNormals&&f.enable(7),y.morphColors&&f.enable(8),y.premultipliedAlpha&&f.enable(9),y.shadowMapEnabled&&f.enable(10),y.doubleSided&&f.enable(11),y.flipSided&&f.enable(12),y.useDepthPacking&&f.enable(13),y.dithering&&f.enable(14),y.transmission&&f.enable(15),y.sheen&&f.enable(16),y.opaque&&f.enable(17),y.pointsUvs&&f.enable(18),y.decodeVideoTexture&&f.enable(19),y.decodeVideoTextureEmissive&&f.enable(20),y.alphaToCoverage&&f.enable(21),D.push(f.mask)}function N(D){const y=R[D.type];let G;if(y){const Y=Gn[y];G=__.clone(Y.uniforms)}else G=D.uniforms;return G}function z(D,y){let G;for(let Y=0,J=_.length;Y<J;Y++){const ee=_[Y];if(ee.cacheKey===y){G=ee,++G.usedTimes;break}}return G===void 0&&(G=new jy(r,y,D,o),_.push(G)),G}function A(D){if(--D.usedTimes===0){const y=_.indexOf(D);_[y]=_[_.length-1],_.pop(),D.destroy()}}function V(D){p.remove(D)}function Z(){p.dispose()}return{getParameters:E,getProgramCacheKey:m,getUniforms:N,acquireProgram:z,releaseProgram:A,releaseShaderCache:V,programs:_,dispose:Z}}function Jy(){let r=new WeakMap;function e(l){return r.has(l)}function t(l){let f=r.get(l);return f===void 0&&(f={},r.set(l,f)),f}function n(l){r.delete(l)}function a(l,f,p){r.get(l)[f]=p}function o(){r=new WeakMap}return{has:e,get:t,remove:n,update:a,dispose:o}}function Qy(r,e){return r.groupOrder!==e.groupOrder?r.groupOrder-e.groupOrder:r.renderOrder!==e.renderOrder?r.renderOrder-e.renderOrder:r.material.id!==e.material.id?r.material.id-e.material.id:r.z!==e.z?r.z-e.z:r.id-e.id}function du(r,e){return r.groupOrder!==e.groupOrder?r.groupOrder-e.groupOrder:r.renderOrder!==e.renderOrder?r.renderOrder-e.renderOrder:r.z!==e.z?e.z-r.z:r.id-e.id}function hu(){const r=[];let e=0;const t=[],n=[],a=[];function o(){e=0,t.length=0,n.length=0,a.length=0}function l(g,v,b,R,P,E){let m=r[e];return m===void 0?(m={id:g.id,object:g,geometry:v,material:b,groupOrder:R,renderOrder:g.renderOrder,z:P,group:E},r[e]=m):(m.id=g.id,m.object=g,m.geometry=v,m.material=b,m.groupOrder=R,m.renderOrder=g.renderOrder,m.z=P,m.group=E),e++,m}function f(g,v,b,R,P,E){const m=l(g,v,b,R,P,E);b.transmission>0?n.push(m):b.transparent===!0?a.push(m):t.push(m)}function p(g,v,b,R,P,E){const m=l(g,v,b,R,P,E);b.transmission>0?n.unshift(m):b.transparent===!0?a.unshift(m):t.unshift(m)}function h(g,v){t.length>1&&t.sort(g||Qy),n.length>1&&n.sort(v||du),a.length>1&&a.sort(v||du)}function _(){for(let g=e,v=r.length;g<v;g++){const b=r[g];if(b.id===null)break;b.id=null,b.object=null,b.geometry=null,b.material=null,b.group=null}}return{opaque:t,transmissive:n,transparent:a,init:o,push:f,unshift:p,finish:_,sort:h}}function eS(){let r=new WeakMap;function e(n,a){const o=r.get(n);let l;return o===void 0?(l=new hu,r.set(n,[l])):a>=o.length?(l=new hu,o.push(l)):l=o[a],l}function t(){r=new WeakMap}return{get:e,dispose:t}}function tS(){const r={};return{get:function(e){if(r[e.id]!==void 0)return r[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new K,color:new xt};break;case"SpotLight":t={position:new K,direction:new K,color:new xt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new K,color:new xt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new K,skyColor:new xt,groundColor:new xt};break;case"RectAreaLight":t={color:new xt,position:new K,halfWidth:new K,halfHeight:new K};break}return r[e.id]=t,t}}}function nS(){const r={};return{get:function(e){if(r[e.id]!==void 0)return r[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new et};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new et};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new et,shadowCameraNear:1,shadowCameraFar:1e3};break}return r[e.id]=t,t}}}let iS=0;function rS(r,e){return(e.castShadow?2:0)-(r.castShadow?2:0)+(e.map?1:0)-(r.map?1:0)}function sS(r){const e=new tS,t=nS(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let h=0;h<9;h++)n.probe.push(new K);const a=new K,o=new Bt,l=new Bt;function f(h){let _=0,g=0,v=0;for(let D=0;D<9;D++)n.probe[D].set(0,0,0);let b=0,R=0,P=0,E=0,m=0,U=0,L=0,N=0,z=0,A=0,V=0;h.sort(rS);for(let D=0,y=h.length;D<y;D++){const G=h[D],Y=G.color,J=G.intensity,ee=G.distance,fe=G.shadow&&G.shadow.map?G.shadow.map.texture:null;if(G.isAmbientLight)_+=Y.r*J,g+=Y.g*J,v+=Y.b*J;else if(G.isLightProbe){for(let le=0;le<9;le++)n.probe[le].addScaledVector(G.sh.coefficients[le],J);V++}else if(G.isDirectionalLight){const le=e.get(G);if(le.color.copy(G.color).multiplyScalar(G.intensity),G.castShadow){const Ee=G.shadow,se=t.get(G);se.shadowIntensity=Ee.intensity,se.shadowBias=Ee.bias,se.shadowNormalBias=Ee.normalBias,se.shadowRadius=Ee.radius,se.shadowMapSize=Ee.mapSize,n.directionalShadow[b]=se,n.directionalShadowMap[b]=fe,n.directionalShadowMatrix[b]=G.shadow.matrix,U++}n.directional[b]=le,b++}else if(G.isSpotLight){const le=e.get(G);le.position.setFromMatrixPosition(G.matrixWorld),le.color.copy(Y).multiplyScalar(J),le.distance=ee,le.coneCos=Math.cos(G.angle),le.penumbraCos=Math.cos(G.angle*(1-G.penumbra)),le.decay=G.decay,n.spot[P]=le;const Ee=G.shadow;if(G.map&&(n.spotLightMap[z]=G.map,z++,Ee.updateMatrices(G),G.castShadow&&A++),n.spotLightMatrix[P]=Ee.matrix,G.castShadow){const se=t.get(G);se.shadowIntensity=Ee.intensity,se.shadowBias=Ee.bias,se.shadowNormalBias=Ee.normalBias,se.shadowRadius=Ee.radius,se.shadowMapSize=Ee.mapSize,n.spotShadow[P]=se,n.spotShadowMap[P]=fe,N++}P++}else if(G.isRectAreaLight){const le=e.get(G);le.color.copy(Y).multiplyScalar(J),le.halfWidth.set(G.width*.5,0,0),le.halfHeight.set(0,G.height*.5,0),n.rectArea[E]=le,E++}else if(G.isPointLight){const le=e.get(G);if(le.color.copy(G.color).multiplyScalar(G.intensity),le.distance=G.distance,le.decay=G.decay,G.castShadow){const Ee=G.shadow,se=t.get(G);se.shadowIntensity=Ee.intensity,se.shadowBias=Ee.bias,se.shadowNormalBias=Ee.normalBias,se.shadowRadius=Ee.radius,se.shadowMapSize=Ee.mapSize,se.shadowCameraNear=Ee.camera.near,se.shadowCameraFar=Ee.camera.far,n.pointShadow[R]=se,n.pointShadowMap[R]=fe,n.pointShadowMatrix[R]=G.shadow.matrix,L++}n.point[R]=le,R++}else if(G.isHemisphereLight){const le=e.get(G);le.skyColor.copy(G.color).multiplyScalar(J),le.groundColor.copy(G.groundColor).multiplyScalar(J),n.hemi[m]=le,m++}}E>0&&(r.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=Ce.LTC_FLOAT_1,n.rectAreaLTC2=Ce.LTC_FLOAT_2):(n.rectAreaLTC1=Ce.LTC_HALF_1,n.rectAreaLTC2=Ce.LTC_HALF_2)),n.ambient[0]=_,n.ambient[1]=g,n.ambient[2]=v;const Z=n.hash;(Z.directionalLength!==b||Z.pointLength!==R||Z.spotLength!==P||Z.rectAreaLength!==E||Z.hemiLength!==m||Z.numDirectionalShadows!==U||Z.numPointShadows!==L||Z.numSpotShadows!==N||Z.numSpotMaps!==z||Z.numLightProbes!==V)&&(n.directional.length=b,n.spot.length=P,n.rectArea.length=E,n.point.length=R,n.hemi.length=m,n.directionalShadow.length=U,n.directionalShadowMap.length=U,n.pointShadow.length=L,n.pointShadowMap.length=L,n.spotShadow.length=N,n.spotShadowMap.length=N,n.directionalShadowMatrix.length=U,n.pointShadowMatrix.length=L,n.spotLightMatrix.length=N+z-A,n.spotLightMap.length=z,n.numSpotLightShadowsWithMaps=A,n.numLightProbes=V,Z.directionalLength=b,Z.pointLength=R,Z.spotLength=P,Z.rectAreaLength=E,Z.hemiLength=m,Z.numDirectionalShadows=U,Z.numPointShadows=L,Z.numSpotShadows=N,Z.numSpotMaps=z,Z.numLightProbes=V,n.version=iS++)}function p(h,_){let g=0,v=0,b=0,R=0,P=0;const E=_.matrixWorldInverse;for(let m=0,U=h.length;m<U;m++){const L=h[m];if(L.isDirectionalLight){const N=n.directional[g];N.direction.setFromMatrixPosition(L.matrixWorld),a.setFromMatrixPosition(L.target.matrixWorld),N.direction.sub(a),N.direction.transformDirection(E),g++}else if(L.isSpotLight){const N=n.spot[b];N.position.setFromMatrixPosition(L.matrixWorld),N.position.applyMatrix4(E),N.direction.setFromMatrixPosition(L.matrixWorld),a.setFromMatrixPosition(L.target.matrixWorld),N.direction.sub(a),N.direction.transformDirection(E),b++}else if(L.isRectAreaLight){const N=n.rectArea[R];N.position.setFromMatrixPosition(L.matrixWorld),N.position.applyMatrix4(E),l.identity(),o.copy(L.matrixWorld),o.premultiply(E),l.extractRotation(o),N.halfWidth.set(L.width*.5,0,0),N.halfHeight.set(0,L.height*.5,0),N.halfWidth.applyMatrix4(l),N.halfHeight.applyMatrix4(l),R++}else if(L.isPointLight){const N=n.point[v];N.position.setFromMatrixPosition(L.matrixWorld),N.position.applyMatrix4(E),v++}else if(L.isHemisphereLight){const N=n.hemi[P];N.direction.setFromMatrixPosition(L.matrixWorld),N.direction.transformDirection(E),P++}}}return{setup:f,setupView:p,state:n}}function pu(r){const e=new sS(r),t=[],n=[];function a(_){h.camera=_,t.length=0,n.length=0}function o(_){t.push(_)}function l(_){n.push(_)}function f(){e.setup(t)}function p(_){e.setupView(t,_)}const h={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:a,state:h,setupLights:f,setupLightsView:p,pushLight:o,pushShadow:l}}function aS(r){let e=new WeakMap;function t(a,o=0){const l=e.get(a);let f;return l===void 0?(f=new pu(r),e.set(a,[f])):o>=l.length?(f=new pu(r),l.push(f)):f=l[o],f}function n(){e=new WeakMap}return{get:t,dispose:n}}const oS=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,cS=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function lS(r,e,t){let n=new mc;const a=new et,o=new et,l=new zt,f=new C_({depthPacking:k0}),p=new P_,h={},_=t.maxTextureSize,g={[Ei]:_n,[_n]:Ei,[Hn]:Hn},v=new ui({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new et},radius:{value:4}},vertexShader:oS,fragmentShader:cS}),b=v.clone();b.defines.HORIZONTAL_PASS=1;const R=new fn;R.setAttribute("position",new An(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const P=new li(R,v),E=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Su;let m=this.type;this.render=function(A,V,Z){if(E.enabled===!1||E.autoUpdate===!1&&E.needsUpdate===!1||A.length===0)return;const D=r.getRenderTarget(),y=r.getActiveCubeFace(),G=r.getActiveMipmapLevel(),Y=r.state;Y.setBlending(ai),Y.buffers.depth.getReversed()===!0?Y.buffers.color.setClear(0,0,0,0):Y.buffers.color.setClear(1,1,1,1),Y.buffers.depth.setTest(!0),Y.setScissorTest(!1);const J=m!==ti&&this.type===ti,ee=m===ti&&this.type!==ti;for(let fe=0,le=A.length;fe<le;fe++){const Ee=A[fe],se=Ee.shadow;if(se===void 0){Ze("WebGLShadowMap:",Ee,"has no shadow.");continue}if(se.autoUpdate===!1&&se.needsUpdate===!1)continue;a.copy(se.mapSize);const Me=se.getFrameExtents();if(a.multiply(Me),o.copy(se.mapSize),(a.x>_||a.y>_)&&(a.x>_&&(o.x=Math.floor(_/Me.x),a.x=o.x*Me.x,se.mapSize.x=o.x),a.y>_&&(o.y=Math.floor(_/Me.y),a.y=o.y*Me.y,se.mapSize.y=o.y)),se.map===null||J===!0||ee===!0){const Ge=this.type!==ti?{minFilter:vn,magFilter:vn}:{};se.map!==null&&se.map.dispose(),se.map=new $i(a.x,a.y,Ge),se.map.texture.name=Ee.name+".shadowMap",se.camera.updateProjectionMatrix()}r.setRenderTarget(se.map),r.clear();const we=se.getViewportCount();for(let Ge=0;Ge<we;Ge++){const ot=se.getViewport(Ge);l.set(o.x*ot.x,o.y*ot.y,o.x*ot.z,o.y*ot.w),Y.viewport(l),se.updateMatrices(Ee,Ge),n=se.getFrustum(),N(V,Z,se.camera,Ee,this.type)}se.isPointLightShadow!==!0&&this.type===ti&&U(se,Z),se.needsUpdate=!1}m=this.type,E.needsUpdate=!1,r.setRenderTarget(D,y,G)};function U(A,V){const Z=e.update(P);v.defines.VSM_SAMPLES!==A.blurSamples&&(v.defines.VSM_SAMPLES=A.blurSamples,b.defines.VSM_SAMPLES=A.blurSamples,v.needsUpdate=!0,b.needsUpdate=!0),A.mapPass===null&&(A.mapPass=new $i(a.x,a.y)),v.uniforms.shadow_pass.value=A.map.texture,v.uniforms.resolution.value=A.mapSize,v.uniforms.radius.value=A.radius,r.setRenderTarget(A.mapPass),r.clear(),r.renderBufferDirect(V,null,Z,v,P,null),b.uniforms.shadow_pass.value=A.mapPass.texture,b.uniforms.resolution.value=A.mapSize,b.uniforms.radius.value=A.radius,r.setRenderTarget(A.map),r.clear(),r.renderBufferDirect(V,null,Z,b,P,null)}function L(A,V,Z,D){let y=null;const G=Z.isPointLight===!0?A.customDistanceMaterial:A.customDepthMaterial;if(G!==void 0)y=G;else if(y=Z.isPointLight===!0?p:f,r.localClippingEnabled&&V.clipShadows===!0&&Array.isArray(V.clippingPlanes)&&V.clippingPlanes.length!==0||V.displacementMap&&V.displacementScale!==0||V.alphaMap&&V.alphaTest>0||V.map&&V.alphaTest>0||V.alphaToCoverage===!0){const Y=y.uuid,J=V.uuid;let ee=h[Y];ee===void 0&&(ee={},h[Y]=ee);let fe=ee[J];fe===void 0&&(fe=y.clone(),ee[J]=fe,V.addEventListener("dispose",z)),y=fe}if(y.visible=V.visible,y.wireframe=V.wireframe,D===ti?y.side=V.shadowSide!==null?V.shadowSide:V.side:y.side=V.shadowSide!==null?V.shadowSide:g[V.side],y.alphaMap=V.alphaMap,y.alphaTest=V.alphaToCoverage===!0?.5:V.alphaTest,y.map=V.map,y.clipShadows=V.clipShadows,y.clippingPlanes=V.clippingPlanes,y.clipIntersection=V.clipIntersection,y.displacementMap=V.displacementMap,y.displacementScale=V.displacementScale,y.displacementBias=V.displacementBias,y.wireframeLinewidth=V.wireframeLinewidth,y.linewidth=V.linewidth,Z.isPointLight===!0&&y.isMeshDistanceMaterial===!0){const Y=r.properties.get(y);Y.light=Z}return y}function N(A,V,Z,D,y){if(A.visible===!1)return;if(A.layers.test(V.layers)&&(A.isMesh||A.isLine||A.isPoints)&&(A.castShadow||A.receiveShadow&&y===ti)&&(!A.frustumCulled||n.intersectsObject(A))){A.modelViewMatrix.multiplyMatrices(Z.matrixWorldInverse,A.matrixWorld);const J=e.update(A),ee=A.material;if(Array.isArray(ee)){const fe=J.groups;for(let le=0,Ee=fe.length;le<Ee;le++){const se=fe[le],Me=ee[se.materialIndex];if(Me&&Me.visible){const we=L(A,Me,D,y);A.onBeforeShadow(r,A,V,Z,J,we,se),r.renderBufferDirect(Z,null,J,we,A,se),A.onAfterShadow(r,A,V,Z,J,we,se)}}}else if(ee.visible){const fe=L(A,ee,D,y);A.onBeforeShadow(r,A,V,Z,J,fe,null),r.renderBufferDirect(Z,null,J,fe,A,null),A.onAfterShadow(r,A,V,Z,J,fe,null)}}const Y=A.children;for(let J=0,ee=Y.length;J<ee;J++)N(Y[J],V,Z,D,y)}function z(A){A.target.removeEventListener("dispose",z);for(const Z in h){const D=h[Z],y=A.target.uuid;y in D&&(D[y].dispose(),delete D[y])}}}const uS={[_o]:xo,[go]:So,[vo]:bo,[br]:yo,[xo]:_o,[So]:go,[bo]:vo,[yo]:br};function fS(r,e){function t(){let k=!1;const Re=new zt;let Te=null;const Se=new zt(0,0,0,0);return{setMask:function(H){Te!==H&&!k&&(r.colorMask(H,H,H,H),Te=H)},setLocked:function(H){k=H},setClear:function(H,B,Le,Ke,vt){vt===!0&&(H*=Ke,B*=Ke,Le*=Ke),Re.set(H,B,Le,Ke),Se.equals(Re)===!1&&(r.clearColor(H,B,Le,Ke),Se.copy(Re))},reset:function(){k=!1,Te=null,Se.set(-1,0,0,0)}}}function n(){let k=!1,Re=!1,Te=null,Se=null,H=null;return{setReversed:function(B){if(Re!==B){const Le=e.get("EXT_clip_control");B?Le.clipControlEXT(Le.LOWER_LEFT_EXT,Le.ZERO_TO_ONE_EXT):Le.clipControlEXT(Le.LOWER_LEFT_EXT,Le.NEGATIVE_ONE_TO_ONE_EXT),Re=B;const Ke=H;H=null,this.setClear(Ke)}},getReversed:function(){return Re},setTest:function(B){B?de(r.DEPTH_TEST):ie(r.DEPTH_TEST)},setMask:function(B){Te!==B&&!k&&(r.depthMask(B),Te=B)},setFunc:function(B){if(Re&&(B=uS[B]),Se!==B){switch(B){case _o:r.depthFunc(r.NEVER);break;case xo:r.depthFunc(r.ALWAYS);break;case go:r.depthFunc(r.LESS);break;case br:r.depthFunc(r.LEQUAL);break;case vo:r.depthFunc(r.EQUAL);break;case yo:r.depthFunc(r.GEQUAL);break;case So:r.depthFunc(r.GREATER);break;case bo:r.depthFunc(r.NOTEQUAL);break;default:r.depthFunc(r.LEQUAL)}Se=B}},setLocked:function(B){k=B},setClear:function(B){H!==B&&(Re&&(B=1-B),r.clearDepth(B),H=B)},reset:function(){k=!1,Te=null,Se=null,H=null,Re=!1}}}function a(){let k=!1,Re=null,Te=null,Se=null,H=null,B=null,Le=null,Ke=null,vt=null;return{setTest:function(ut){k||(ut?de(r.STENCIL_TEST):ie(r.STENCIL_TEST))},setMask:function(ut){Re!==ut&&!k&&(r.stencilMask(ut),Re=ut)},setFunc:function(ut,yn,Vt){(Te!==ut||Se!==yn||H!==Vt)&&(r.stencilFunc(ut,yn,Vt),Te=ut,Se=yn,H=Vt)},setOp:function(ut,yn,Vt){(B!==ut||Le!==yn||Ke!==Vt)&&(r.stencilOp(ut,yn,Vt),B=ut,Le=yn,Ke=Vt)},setLocked:function(ut){k=ut},setClear:function(ut){vt!==ut&&(r.clearStencil(ut),vt=ut)},reset:function(){k=!1,Re=null,Te=null,Se=null,H=null,B=null,Le=null,Ke=null,vt=null}}}const o=new t,l=new n,f=new a,p=new WeakMap,h=new WeakMap;let _={},g={},v=new WeakMap,b=[],R=null,P=!1,E=null,m=null,U=null,L=null,N=null,z=null,A=null,V=new xt(0,0,0),Z=0,D=!1,y=null,G=null,Y=null,J=null,ee=null;const fe=r.getParameter(r.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let le=!1,Ee=0;const se=r.getParameter(r.VERSION);se.indexOf("WebGL")!==-1?(Ee=parseFloat(/^WebGL (\d)/.exec(se)[1]),le=Ee>=1):se.indexOf("OpenGL ES")!==-1&&(Ee=parseFloat(/^OpenGL ES (\d)/.exec(se)[1]),le=Ee>=2);let Me=null,we={};const Ge=r.getParameter(r.SCISSOR_BOX),ot=r.getParameter(r.VIEWPORT),_t=new zt().fromArray(Ge),Je=new zt().fromArray(ot);function at(k,Re,Te,Se){const H=new Uint8Array(4),B=r.createTexture();r.bindTexture(k,B),r.texParameteri(k,r.TEXTURE_MIN_FILTER,r.NEAREST),r.texParameteri(k,r.TEXTURE_MAG_FILTER,r.NEAREST);for(let Le=0;Le<Te;Le++)k===r.TEXTURE_3D||k===r.TEXTURE_2D_ARRAY?r.texImage3D(Re,0,r.RGBA,1,1,Se,0,r.RGBA,r.UNSIGNED_BYTE,H):r.texImage2D(Re+Le,0,r.RGBA,1,1,0,r.RGBA,r.UNSIGNED_BYTE,H);return B}const ce={};ce[r.TEXTURE_2D]=at(r.TEXTURE_2D,r.TEXTURE_2D,1),ce[r.TEXTURE_CUBE_MAP]=at(r.TEXTURE_CUBE_MAP,r.TEXTURE_CUBE_MAP_POSITIVE_X,6),ce[r.TEXTURE_2D_ARRAY]=at(r.TEXTURE_2D_ARRAY,r.TEXTURE_2D_ARRAY,1,1),ce[r.TEXTURE_3D]=at(r.TEXTURE_3D,r.TEXTURE_3D,1,1),o.setClear(0,0,0,1),l.setClear(1),f.setClear(0),de(r.DEPTH_TEST),l.setFunc(br),ct(!1),lt(hl),de(r.CULL_FACE),Mt(ai);function de(k){_[k]!==!0&&(r.enable(k),_[k]=!0)}function ie(k){_[k]!==!1&&(r.disable(k),_[k]=!1)}function ye(k,Re){return g[k]!==Re?(r.bindFramebuffer(k,Re),g[k]=Re,k===r.DRAW_FRAMEBUFFER&&(g[r.FRAMEBUFFER]=Re),k===r.FRAMEBUFFER&&(g[r.DRAW_FRAMEBUFFER]=Re),!0):!1}function Be(k,Re){let Te=b,Se=!1;if(k){Te=v.get(Re),Te===void 0&&(Te=[],v.set(Re,Te));const H=k.textures;if(Te.length!==H.length||Te[0]!==r.COLOR_ATTACHMENT0){for(let B=0,Le=H.length;B<Le;B++)Te[B]=r.COLOR_ATTACHMENT0+B;Te.length=H.length,Se=!0}}else Te[0]!==r.BACK&&(Te[0]=r.BACK,Se=!0);Se&&r.drawBuffers(Te)}function nt(k){return R!==k?(r.useProgram(k),R=k,!0):!1}const Pt={[Vi]:r.FUNC_ADD,[u0]:r.FUNC_SUBTRACT,[f0]:r.FUNC_REVERSE_SUBTRACT};Pt[d0]=r.MIN,Pt[h0]=r.MAX;const it={[p0]:r.ZERO,[m0]:r.ONE,[_0]:r.SRC_COLOR,[po]:r.SRC_ALPHA,[b0]:r.SRC_ALPHA_SATURATE,[y0]:r.DST_COLOR,[g0]:r.DST_ALPHA,[x0]:r.ONE_MINUS_SRC_COLOR,[mo]:r.ONE_MINUS_SRC_ALPHA,[S0]:r.ONE_MINUS_DST_COLOR,[v0]:r.ONE_MINUS_DST_ALPHA,[E0]:r.CONSTANT_COLOR,[M0]:r.ONE_MINUS_CONSTANT_COLOR,[T0]:r.CONSTANT_ALPHA,[w0]:r.ONE_MINUS_CONSTANT_ALPHA};function Mt(k,Re,Te,Se,H,B,Le,Ke,vt,ut){if(k===ai){P===!0&&(ie(r.BLEND),P=!1);return}if(P===!1&&(de(r.BLEND),P=!0),k!==l0){if(k!==E||ut!==D){if((m!==Vi||N!==Vi)&&(r.blendEquation(r.FUNC_ADD),m=Vi,N=Vi),ut)switch(k){case gr:r.blendFuncSeparate(r.ONE,r.ONE_MINUS_SRC_ALPHA,r.ONE,r.ONE_MINUS_SRC_ALPHA);break;case pl:r.blendFunc(r.ONE,r.ONE);break;case ml:r.blendFuncSeparate(r.ZERO,r.ONE_MINUS_SRC_COLOR,r.ZERO,r.ONE);break;case _l:r.blendFuncSeparate(r.DST_COLOR,r.ONE_MINUS_SRC_ALPHA,r.ZERO,r.ONE);break;default:Gt("WebGLState: Invalid blending: ",k);break}else switch(k){case gr:r.blendFuncSeparate(r.SRC_ALPHA,r.ONE_MINUS_SRC_ALPHA,r.ONE,r.ONE_MINUS_SRC_ALPHA);break;case pl:r.blendFuncSeparate(r.SRC_ALPHA,r.ONE,r.ONE,r.ONE);break;case ml:Gt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case _l:Gt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Gt("WebGLState: Invalid blending: ",k);break}U=null,L=null,z=null,A=null,V.set(0,0,0),Z=0,E=k,D=ut}return}H=H||Re,B=B||Te,Le=Le||Se,(Re!==m||H!==N)&&(r.blendEquationSeparate(Pt[Re],Pt[H]),m=Re,N=H),(Te!==U||Se!==L||B!==z||Le!==A)&&(r.blendFuncSeparate(it[Te],it[Se],it[B],it[Le]),U=Te,L=Se,z=B,A=Le),(Ke.equals(V)===!1||vt!==Z)&&(r.blendColor(Ke.r,Ke.g,Ke.b,vt),V.copy(Ke),Z=vt),E=k,D=!1}function O(k,Re){k.side===Hn?ie(r.CULL_FACE):de(r.CULL_FACE);let Te=k.side===_n;Re&&(Te=!Te),ct(Te),k.blending===gr&&k.transparent===!1?Mt(ai):Mt(k.blending,k.blendEquation,k.blendSrc,k.blendDst,k.blendEquationAlpha,k.blendSrcAlpha,k.blendDstAlpha,k.blendColor,k.blendAlpha,k.premultipliedAlpha),l.setFunc(k.depthFunc),l.setTest(k.depthTest),l.setMask(k.depthWrite),o.setMask(k.colorWrite);const Se=k.stencilWrite;f.setTest(Se),Se&&(f.setMask(k.stencilWriteMask),f.setFunc(k.stencilFunc,k.stencilRef,k.stencilFuncMask),f.setOp(k.stencilFail,k.stencilZFail,k.stencilZPass)),Ie(k.polygonOffset,k.polygonOffsetFactor,k.polygonOffsetUnits),k.alphaToCoverage===!0?de(r.SAMPLE_ALPHA_TO_COVERAGE):ie(r.SAMPLE_ALPHA_TO_COVERAGE)}function ct(k){y!==k&&(k?r.frontFace(r.CW):r.frontFace(r.CCW),y=k)}function lt(k){k!==a0?(de(r.CULL_FACE),k!==G&&(k===hl?r.cullFace(r.BACK):k===o0?r.cullFace(r.FRONT):r.cullFace(r.FRONT_AND_BACK))):ie(r.CULL_FACE),G=k}function Dt(k){k!==Y&&(le&&r.lineWidth(k),Y=k)}function Ie(k,Re,Te){k?(de(r.POLYGON_OFFSET_FILL),(J!==Re||ee!==Te)&&(r.polygonOffset(Re,Te),J=Re,ee=Te)):ie(r.POLYGON_OFFSET_FILL)}function Rt(k){k?de(r.SCISSOR_TEST):ie(r.SCISSOR_TEST)}function Ne(k){k===void 0&&(k=r.TEXTURE0+fe-1),Me!==k&&(r.activeTexture(k),Me=k)}function $e(k,Re,Te){Te===void 0&&(Me===null?Te=r.TEXTURE0+fe-1:Te=Me);let Se=we[Te];Se===void 0&&(Se={type:void 0,texture:void 0},we[Te]=Se),(Se.type!==k||Se.texture!==Re)&&(Me!==Te&&(r.activeTexture(Te),Me=Te),r.bindTexture(k,Re||ce[k]),Se.type=k,Se.texture=Re)}function F(){const k=we[Me];k!==void 0&&k.type!==void 0&&(r.bindTexture(k.type,null),k.type=void 0,k.texture=void 0)}function w(){try{r.compressedTexImage2D(...arguments)}catch(k){k("WebGLState:",k)}}function $(){try{r.compressedTexImage3D(...arguments)}catch(k){k("WebGLState:",k)}}function re(){try{r.texSubImage2D(...arguments)}catch(k){k("WebGLState:",k)}}function he(){try{r.texSubImage3D(...arguments)}catch(k){k("WebGLState:",k)}}function ne(){try{r.compressedTexSubImage2D(...arguments)}catch(k){k("WebGLState:",k)}}function ke(){try{r.compressedTexSubImage3D(...arguments)}catch(k){k("WebGLState:",k)}}function Ae(){try{r.texStorage2D(...arguments)}catch(k){k("WebGLState:",k)}}function He(){try{r.texStorage3D(...arguments)}catch(k){k("WebGLState:",k)}}function Oe(){try{r.texImage2D(...arguments)}catch(k){k("WebGLState:",k)}}function me(){try{r.texImage3D(...arguments)}catch(k){k("WebGLState:",k)}}function be(k){_t.equals(k)===!1&&(r.scissor(k.x,k.y,k.z,k.w),_t.copy(k))}function je(k){Je.equals(k)===!1&&(r.viewport(k.x,k.y,k.z,k.w),Je.copy(k))}function We(k,Re){let Te=h.get(Re);Te===void 0&&(Te=new WeakMap,h.set(Re,Te));let Se=Te.get(k);Se===void 0&&(Se=r.getUniformBlockIndex(Re,k.name),Te.set(k,Se))}function De(k,Re){const Se=h.get(Re).get(k);p.get(Re)!==Se&&(r.uniformBlockBinding(Re,Se,k.__bindingPointIndex),p.set(Re,Se))}function qe(){r.disable(r.BLEND),r.disable(r.CULL_FACE),r.disable(r.DEPTH_TEST),r.disable(r.POLYGON_OFFSET_FILL),r.disable(r.SCISSOR_TEST),r.disable(r.STENCIL_TEST),r.disable(r.SAMPLE_ALPHA_TO_COVERAGE),r.blendEquation(r.FUNC_ADD),r.blendFunc(r.ONE,r.ZERO),r.blendFuncSeparate(r.ONE,r.ZERO,r.ONE,r.ZERO),r.blendColor(0,0,0,0),r.colorMask(!0,!0,!0,!0),r.clearColor(0,0,0,0),r.depthMask(!0),r.depthFunc(r.LESS),l.setReversed(!1),r.clearDepth(1),r.stencilMask(4294967295),r.stencilFunc(r.ALWAYS,0,4294967295),r.stencilOp(r.KEEP,r.KEEP,r.KEEP),r.clearStencil(0),r.cullFace(r.BACK),r.frontFace(r.CCW),r.polygonOffset(0,0),r.activeTexture(r.TEXTURE0),r.bindFramebuffer(r.FRAMEBUFFER,null),r.bindFramebuffer(r.DRAW_FRAMEBUFFER,null),r.bindFramebuffer(r.READ_FRAMEBUFFER,null),r.useProgram(null),r.lineWidth(1),r.scissor(0,0,r.canvas.width,r.canvas.height),r.viewport(0,0,r.canvas.width,r.canvas.height),_={},Me=null,we={},g={},v=new WeakMap,b=[],R=null,P=!1,E=null,m=null,U=null,L=null,N=null,z=null,A=null,V=new xt(0,0,0),Z=0,D=!1,y=null,G=null,Y=null,J=null,ee=null,_t.set(0,0,r.canvas.width,r.canvas.height),Je.set(0,0,r.canvas.width,r.canvas.height),o.reset(),l.reset(),f.reset()}return{buffers:{color:o,depth:l,stencil:f},enable:de,disable:ie,bindFramebuffer:ye,drawBuffers:Be,useProgram:nt,setBlending:Mt,setMaterial:O,setFlipSided:ct,setCullFace:lt,setLineWidth:Dt,setPolygonOffset:Ie,setScissorTest:Rt,activeTexture:Ne,bindTexture:$e,unbindTexture:F,compressedTexImage2D:w,compressedTexImage3D:$,texImage2D:Oe,texImage3D:me,updateUBOMapping:We,uniformBlockBinding:De,texStorage2D:Ae,texStorage3D:He,texSubImage2D:re,texSubImage3D:he,compressedTexSubImage2D:ne,compressedTexSubImage3D:ke,scissor:be,viewport:je,reset:qe}}function dS(r,e,t,n,a,o,l){const f=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,p=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),h=new et,_=new WeakMap;let g;const v=new WeakMap;let b=!1;try{b=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function R(F,w){return b?new OffscreenCanvas(F,w):sa("canvas")}function P(F,w,$){let re=1;const he=$e(F);if((he.width>$||he.height>$)&&(re=$/Math.max(he.width,he.height)),re<1)if(typeof HTMLImageElement<"u"&&F instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&F instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&F instanceof ImageBitmap||typeof VideoFrame<"u"&&F instanceof VideoFrame){const ne=Math.floor(re*he.width),ke=Math.floor(re*he.height);g===void 0&&(g=R(ne,ke));const Ae=w?R(ne,ke):g;return Ae.width=ne,Ae.height=ke,Ae.getContext("2d").drawImage(F,0,0,ne,ke),Ze("WebGLRenderer: Texture has been resized from ("+he.width+"x"+he.height+") to ("+ne+"x"+ke+")."),Ae}else return"data"in F&&Ze("WebGLRenderer: Image in DataTexture is too big ("+he.width+"x"+he.height+")."),F;return F}function E(F){return F.generateMipmaps}function m(F){r.generateMipmap(F)}function U(F){return F.isWebGLCubeRenderTarget?r.TEXTURE_CUBE_MAP:F.isWebGL3DRenderTarget?r.TEXTURE_3D:F.isWebGLArrayRenderTarget||F.isCompressedArrayTexture?r.TEXTURE_2D_ARRAY:r.TEXTURE_2D}function L(F,w,$,re,he=!1){if(F!==null){if(r[F]!==void 0)return r[F];Ze("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+F+"'")}let ne=w;if(w===r.RED&&($===r.FLOAT&&(ne=r.R32F),$===r.HALF_FLOAT&&(ne=r.R16F),$===r.UNSIGNED_BYTE&&(ne=r.R8)),w===r.RED_INTEGER&&($===r.UNSIGNED_BYTE&&(ne=r.R8UI),$===r.UNSIGNED_SHORT&&(ne=r.R16UI),$===r.UNSIGNED_INT&&(ne=r.R32UI),$===r.BYTE&&(ne=r.R8I),$===r.SHORT&&(ne=r.R16I),$===r.INT&&(ne=r.R32I)),w===r.RG&&($===r.FLOAT&&(ne=r.RG32F),$===r.HALF_FLOAT&&(ne=r.RG16F),$===r.UNSIGNED_BYTE&&(ne=r.RG8)),w===r.RG_INTEGER&&($===r.UNSIGNED_BYTE&&(ne=r.RG8UI),$===r.UNSIGNED_SHORT&&(ne=r.RG16UI),$===r.UNSIGNED_INT&&(ne=r.RG32UI),$===r.BYTE&&(ne=r.RG8I),$===r.SHORT&&(ne=r.RG16I),$===r.INT&&(ne=r.RG32I)),w===r.RGB_INTEGER&&($===r.UNSIGNED_BYTE&&(ne=r.RGB8UI),$===r.UNSIGNED_SHORT&&(ne=r.RGB16UI),$===r.UNSIGNED_INT&&(ne=r.RGB32UI),$===r.BYTE&&(ne=r.RGB8I),$===r.SHORT&&(ne=r.RGB16I),$===r.INT&&(ne=r.RGB32I)),w===r.RGBA_INTEGER&&($===r.UNSIGNED_BYTE&&(ne=r.RGBA8UI),$===r.UNSIGNED_SHORT&&(ne=r.RGBA16UI),$===r.UNSIGNED_INT&&(ne=r.RGBA32UI),$===r.BYTE&&(ne=r.RGBA8I),$===r.SHORT&&(ne=r.RGBA16I),$===r.INT&&(ne=r.RGBA32I)),w===r.RGB&&($===r.UNSIGNED_INT_5_9_9_9_REV&&(ne=r.RGB9_E5),$===r.UNSIGNED_INT_10F_11F_11F_REV&&(ne=r.R11F_G11F_B10F)),w===r.RGBA){const ke=he?ia:yt.getTransfer(re);$===r.FLOAT&&(ne=r.RGBA32F),$===r.HALF_FLOAT&&(ne=r.RGBA16F),$===r.UNSIGNED_BYTE&&(ne=ke===Ft?r.SRGB8_ALPHA8:r.RGBA8),$===r.UNSIGNED_SHORT_4_4_4_4&&(ne=r.RGBA4),$===r.UNSIGNED_SHORT_5_5_5_1&&(ne=r.RGB5_A1)}return(ne===r.R16F||ne===r.R32F||ne===r.RG16F||ne===r.RG32F||ne===r.RGBA16F||ne===r.RGBA32F)&&e.get("EXT_color_buffer_float"),ne}function N(F,w){let $;return F?w===null||w===Wi||w===Yr?$=r.DEPTH24_STENCIL8:w===ii?$=r.DEPTH32F_STENCIL8:w===jr&&($=r.DEPTH24_STENCIL8,Ze("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):w===null||w===Wi||w===Yr?$=r.DEPTH_COMPONENT24:w===ii?$=r.DEPTH_COMPONENT32F:w===jr&&($=r.DEPTH_COMPONENT16),$}function z(F,w){return E(F)===!0||F.isFramebufferTexture&&F.minFilter!==vn&&F.minFilter!==wn?Math.log2(Math.max(w.width,w.height))+1:F.mipmaps!==void 0&&F.mipmaps.length>0?F.mipmaps.length:F.isCompressedTexture&&Array.isArray(F.image)?w.mipmaps.length:1}function A(F){const w=F.target;w.removeEventListener("dispose",A),Z(w),w.isVideoTexture&&_.delete(w)}function V(F){const w=F.target;w.removeEventListener("dispose",V),y(w)}function Z(F){const w=n.get(F);if(w.__webglInit===void 0)return;const $=F.source,re=v.get($);if(re){const he=re[w.__cacheKey];he.usedTimes--,he.usedTimes===0&&D(F),Object.keys(re).length===0&&v.delete($)}n.remove(F)}function D(F){const w=n.get(F);r.deleteTexture(w.__webglTexture);const $=F.source,re=v.get($);delete re[w.__cacheKey],l.memory.textures--}function y(F){const w=n.get(F);if(F.depthTexture&&(F.depthTexture.dispose(),n.remove(F.depthTexture)),F.isWebGLCubeRenderTarget)for(let re=0;re<6;re++){if(Array.isArray(w.__webglFramebuffer[re]))for(let he=0;he<w.__webglFramebuffer[re].length;he++)r.deleteFramebuffer(w.__webglFramebuffer[re][he]);else r.deleteFramebuffer(w.__webglFramebuffer[re]);w.__webglDepthbuffer&&r.deleteRenderbuffer(w.__webglDepthbuffer[re])}else{if(Array.isArray(w.__webglFramebuffer))for(let re=0;re<w.__webglFramebuffer.length;re++)r.deleteFramebuffer(w.__webglFramebuffer[re]);else r.deleteFramebuffer(w.__webglFramebuffer);if(w.__webglDepthbuffer&&r.deleteRenderbuffer(w.__webglDepthbuffer),w.__webglMultisampledFramebuffer&&r.deleteFramebuffer(w.__webglMultisampledFramebuffer),w.__webglColorRenderbuffer)for(let re=0;re<w.__webglColorRenderbuffer.length;re++)w.__webglColorRenderbuffer[re]&&r.deleteRenderbuffer(w.__webglColorRenderbuffer[re]);w.__webglDepthRenderbuffer&&r.deleteRenderbuffer(w.__webglDepthRenderbuffer)}const $=F.textures;for(let re=0,he=$.length;re<he;re++){const ne=n.get($[re]);ne.__webglTexture&&(r.deleteTexture(ne.__webglTexture),l.memory.textures--),n.remove($[re])}n.remove(F)}let G=0;function Y(){G=0}function J(){const F=G;return F>=a.maxTextures&&Ze("WebGLTextures: Trying to use "+F+" texture units while this GPU supports only "+a.maxTextures),G+=1,F}function ee(F){const w=[];return w.push(F.wrapS),w.push(F.wrapT),w.push(F.wrapR||0),w.push(F.magFilter),w.push(F.minFilter),w.push(F.anisotropy),w.push(F.internalFormat),w.push(F.format),w.push(F.type),w.push(F.generateMipmaps),w.push(F.premultiplyAlpha),w.push(F.flipY),w.push(F.unpackAlignment),w.push(F.colorSpace),w.join()}function fe(F,w){const $=n.get(F);if(F.isVideoTexture&&Rt(F),F.isRenderTargetTexture===!1&&F.isExternalTexture!==!0&&F.version>0&&$.__version!==F.version){const re=F.image;if(re===null)Ze("WebGLRenderer: Texture marked for update but no image data found.");else if(re.complete===!1)Ze("WebGLRenderer: Texture marked for update but image is incomplete");else{ce($,F,w);return}}else F.isExternalTexture&&($.__webglTexture=F.sourceTexture?F.sourceTexture:null);t.bindTexture(r.TEXTURE_2D,$.__webglTexture,r.TEXTURE0+w)}function le(F,w){const $=n.get(F);if(F.isRenderTargetTexture===!1&&F.version>0&&$.__version!==F.version){ce($,F,w);return}else F.isExternalTexture&&($.__webglTexture=F.sourceTexture?F.sourceTexture:null);t.bindTexture(r.TEXTURE_2D_ARRAY,$.__webglTexture,r.TEXTURE0+w)}function Ee(F,w){const $=n.get(F);if(F.isRenderTargetTexture===!1&&F.version>0&&$.__version!==F.version){ce($,F,w);return}t.bindTexture(r.TEXTURE_3D,$.__webglTexture,r.TEXTURE0+w)}function se(F,w){const $=n.get(F);if(F.version>0&&$.__version!==F.version){de($,F,w);return}t.bindTexture(r.TEXTURE_CUBE_MAP,$.__webglTexture,r.TEXTURE0+w)}const Me={[To]:r.REPEAT,[ni]:r.CLAMP_TO_EDGE,[wo]:r.MIRRORED_REPEAT},we={[vn]:r.NEAREST,[N0]:r.NEAREST_MIPMAP_NEAREST,[Ts]:r.NEAREST_MIPMAP_LINEAR,[wn]:r.LINEAR,[Ia]:r.LINEAR_MIPMAP_NEAREST,[Hi]:r.LINEAR_MIPMAP_LINEAR},Ge={[z0]:r.NEVER,[$0]:r.ALWAYS,[V0]:r.LESS,[Lu]:r.LEQUAL,[G0]:r.EQUAL,[X0]:r.GEQUAL,[H0]:r.GREATER,[W0]:r.NOTEQUAL};function ot(F,w){if(w.type===ii&&e.has("OES_texture_float_linear")===!1&&(w.magFilter===wn||w.magFilter===Ia||w.magFilter===Ts||w.magFilter===Hi||w.minFilter===wn||w.minFilter===Ia||w.minFilter===Ts||w.minFilter===Hi)&&Ze("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),r.texParameteri(F,r.TEXTURE_WRAP_S,Me[w.wrapS]),r.texParameteri(F,r.TEXTURE_WRAP_T,Me[w.wrapT]),(F===r.TEXTURE_3D||F===r.TEXTURE_2D_ARRAY)&&r.texParameteri(F,r.TEXTURE_WRAP_R,Me[w.wrapR]),r.texParameteri(F,r.TEXTURE_MAG_FILTER,we[w.magFilter]),r.texParameteri(F,r.TEXTURE_MIN_FILTER,we[w.minFilter]),w.compareFunction&&(r.texParameteri(F,r.TEXTURE_COMPARE_MODE,r.COMPARE_REF_TO_TEXTURE),r.texParameteri(F,r.TEXTURE_COMPARE_FUNC,Ge[w.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(w.magFilter===vn||w.minFilter!==Ts&&w.minFilter!==Hi||w.type===ii&&e.has("OES_texture_float_linear")===!1)return;if(w.anisotropy>1||n.get(w).__currentAnisotropy){const $=e.get("EXT_texture_filter_anisotropic");r.texParameterf(F,$.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(w.anisotropy,a.getMaxAnisotropy())),n.get(w).__currentAnisotropy=w.anisotropy}}}function _t(F,w){let $=!1;F.__webglInit===void 0&&(F.__webglInit=!0,w.addEventListener("dispose",A));const re=w.source;let he=v.get(re);he===void 0&&(he={},v.set(re,he));const ne=ee(w);if(ne!==F.__cacheKey){he[ne]===void 0&&(he[ne]={texture:r.createTexture(),usedTimes:0},l.memory.textures++,$=!0),he[ne].usedTimes++;const ke=he[F.__cacheKey];ke!==void 0&&(he[F.__cacheKey].usedTimes--,ke.usedTimes===0&&D(w)),F.__cacheKey=ne,F.__webglTexture=he[ne].texture}return $}function Je(F,w,$){return Math.floor(Math.floor(F/$)/w)}function at(F,w,$,re){const ne=F.updateRanges;if(ne.length===0)t.texSubImage2D(r.TEXTURE_2D,0,0,0,w.width,w.height,$,re,w.data);else{ne.sort((me,be)=>me.start-be.start);let ke=0;for(let me=1;me<ne.length;me++){const be=ne[ke],je=ne[me],We=be.start+be.count,De=Je(je.start,w.width,4),qe=Je(be.start,w.width,4);je.start<=We+1&&De===qe&&Je(je.start+je.count-1,w.width,4)===De?be.count=Math.max(be.count,je.start+je.count-be.start):(++ke,ne[ke]=je)}ne.length=ke+1;const Ae=r.getParameter(r.UNPACK_ROW_LENGTH),He=r.getParameter(r.UNPACK_SKIP_PIXELS),Oe=r.getParameter(r.UNPACK_SKIP_ROWS);r.pixelStorei(r.UNPACK_ROW_LENGTH,w.width);for(let me=0,be=ne.length;me<be;me++){const je=ne[me],We=Math.floor(je.start/4),De=Math.ceil(je.count/4),qe=We%w.width,k=Math.floor(We/w.width),Re=De,Te=1;r.pixelStorei(r.UNPACK_SKIP_PIXELS,qe),r.pixelStorei(r.UNPACK_SKIP_ROWS,k),t.texSubImage2D(r.TEXTURE_2D,0,qe,k,Re,Te,$,re,w.data)}F.clearUpdateRanges(),r.pixelStorei(r.UNPACK_ROW_LENGTH,Ae),r.pixelStorei(r.UNPACK_SKIP_PIXELS,He),r.pixelStorei(r.UNPACK_SKIP_ROWS,Oe)}}function ce(F,w,$){let re=r.TEXTURE_2D;(w.isDataArrayTexture||w.isCompressedArrayTexture)&&(re=r.TEXTURE_2D_ARRAY),w.isData3DTexture&&(re=r.TEXTURE_3D);const he=_t(F,w),ne=w.source;t.bindTexture(re,F.__webglTexture,r.TEXTURE0+$);const ke=n.get(ne);if(ne.version!==ke.__version||he===!0){t.activeTexture(r.TEXTURE0+$);const Ae=yt.getPrimaries(yt.workingColorSpace),He=w.colorSpace===yi?null:yt.getPrimaries(w.colorSpace),Oe=w.colorSpace===yi||Ae===He?r.NONE:r.BROWSER_DEFAULT_WEBGL;r.pixelStorei(r.UNPACK_FLIP_Y_WEBGL,w.flipY),r.pixelStorei(r.UNPACK_PREMULTIPLY_ALPHA_WEBGL,w.premultiplyAlpha),r.pixelStorei(r.UNPACK_ALIGNMENT,w.unpackAlignment),r.pixelStorei(r.UNPACK_COLORSPACE_CONVERSION_WEBGL,Oe);let me=P(w.image,!1,a.maxTextureSize);me=Ne(w,me);const be=o.convert(w.format,w.colorSpace),je=o.convert(w.type);let We=L(w.internalFormat,be,je,w.colorSpace,w.isVideoTexture);ot(re,w);let De;const qe=w.mipmaps,k=w.isVideoTexture!==!0,Re=ke.__version===void 0||he===!0,Te=ne.dataReady,Se=z(w,me);if(w.isDepthTexture)We=N(w.format===Kr,w.type),Re&&(k?t.texStorage2D(r.TEXTURE_2D,1,We,me.width,me.height):t.texImage2D(r.TEXTURE_2D,0,We,me.width,me.height,0,be,je,null));else if(w.isDataTexture)if(qe.length>0){k&&Re&&t.texStorage2D(r.TEXTURE_2D,Se,We,qe[0].width,qe[0].height);for(let H=0,B=qe.length;H<B;H++)De=qe[H],k?Te&&t.texSubImage2D(r.TEXTURE_2D,H,0,0,De.width,De.height,be,je,De.data):t.texImage2D(r.TEXTURE_2D,H,We,De.width,De.height,0,be,je,De.data);w.generateMipmaps=!1}else k?(Re&&t.texStorage2D(r.TEXTURE_2D,Se,We,me.width,me.height),Te&&at(w,me,be,je)):t.texImage2D(r.TEXTURE_2D,0,We,me.width,me.height,0,be,je,me.data);else if(w.isCompressedTexture)if(w.isCompressedArrayTexture){k&&Re&&t.texStorage3D(r.TEXTURE_2D_ARRAY,Se,We,qe[0].width,qe[0].height,me.depth);for(let H=0,B=qe.length;H<B;H++)if(De=qe[H],w.format!==Nn)if(be!==null)if(k){if(Te)if(w.layerUpdates.size>0){const Le=Xl(De.width,De.height,w.format,w.type);for(const Ke of w.layerUpdates){const vt=De.data.subarray(Ke*Le/De.data.BYTES_PER_ELEMENT,(Ke+1)*Le/De.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(r.TEXTURE_2D_ARRAY,H,0,0,Ke,De.width,De.height,1,be,vt)}w.clearLayerUpdates()}else t.compressedTexSubImage3D(r.TEXTURE_2D_ARRAY,H,0,0,0,De.width,De.height,me.depth,be,De.data)}else t.compressedTexImage3D(r.TEXTURE_2D_ARRAY,H,We,De.width,De.height,me.depth,0,De.data,0,0);else Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else k?Te&&t.texSubImage3D(r.TEXTURE_2D_ARRAY,H,0,0,0,De.width,De.height,me.depth,be,je,De.data):t.texImage3D(r.TEXTURE_2D_ARRAY,H,We,De.width,De.height,me.depth,0,be,je,De.data)}else{k&&Re&&t.texStorage2D(r.TEXTURE_2D,Se,We,qe[0].width,qe[0].height);for(let H=0,B=qe.length;H<B;H++)De=qe[H],w.format!==Nn?be!==null?k?Te&&t.compressedTexSubImage2D(r.TEXTURE_2D,H,0,0,De.width,De.height,be,De.data):t.compressedTexImage2D(r.TEXTURE_2D,H,We,De.width,De.height,0,De.data):Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):k?Te&&t.texSubImage2D(r.TEXTURE_2D,H,0,0,De.width,De.height,be,je,De.data):t.texImage2D(r.TEXTURE_2D,H,We,De.width,De.height,0,be,je,De.data)}else if(w.isDataArrayTexture)if(k){if(Re&&t.texStorage3D(r.TEXTURE_2D_ARRAY,Se,We,me.width,me.height,me.depth),Te)if(w.layerUpdates.size>0){const H=Xl(me.width,me.height,w.format,w.type);for(const B of w.layerUpdates){const Le=me.data.subarray(B*H/me.data.BYTES_PER_ELEMENT,(B+1)*H/me.data.BYTES_PER_ELEMENT);t.texSubImage3D(r.TEXTURE_2D_ARRAY,0,0,0,B,me.width,me.height,1,be,je,Le)}w.clearLayerUpdates()}else t.texSubImage3D(r.TEXTURE_2D_ARRAY,0,0,0,0,me.width,me.height,me.depth,be,je,me.data)}else t.texImage3D(r.TEXTURE_2D_ARRAY,0,We,me.width,me.height,me.depth,0,be,je,me.data);else if(w.isData3DTexture)k?(Re&&t.texStorage3D(r.TEXTURE_3D,Se,We,me.width,me.height,me.depth),Te&&t.texSubImage3D(r.TEXTURE_3D,0,0,0,0,me.width,me.height,me.depth,be,je,me.data)):t.texImage3D(r.TEXTURE_3D,0,We,me.width,me.height,me.depth,0,be,je,me.data);else if(w.isFramebufferTexture){if(Re)if(k)t.texStorage2D(r.TEXTURE_2D,Se,We,me.width,me.height);else{let H=me.width,B=me.height;for(let Le=0;Le<Se;Le++)t.texImage2D(r.TEXTURE_2D,Le,We,H,B,0,be,je,null),H>>=1,B>>=1}}else if(qe.length>0){if(k&&Re){const H=$e(qe[0]);t.texStorage2D(r.TEXTURE_2D,Se,We,H.width,H.height)}for(let H=0,B=qe.length;H<B;H++)De=qe[H],k?Te&&t.texSubImage2D(r.TEXTURE_2D,H,0,0,be,je,De):t.texImage2D(r.TEXTURE_2D,H,We,be,je,De);w.generateMipmaps=!1}else if(k){if(Re){const H=$e(me);t.texStorage2D(r.TEXTURE_2D,Se,We,H.width,H.height)}Te&&t.texSubImage2D(r.TEXTURE_2D,0,0,0,be,je,me)}else t.texImage2D(r.TEXTURE_2D,0,We,be,je,me);E(w)&&m(re),ke.__version=ne.version,w.onUpdate&&w.onUpdate(w)}F.__version=w.version}function de(F,w,$){if(w.image.length!==6)return;const re=_t(F,w),he=w.source;t.bindTexture(r.TEXTURE_CUBE_MAP,F.__webglTexture,r.TEXTURE0+$);const ne=n.get(he);if(he.version!==ne.__version||re===!0){t.activeTexture(r.TEXTURE0+$);const ke=yt.getPrimaries(yt.workingColorSpace),Ae=w.colorSpace===yi?null:yt.getPrimaries(w.colorSpace),He=w.colorSpace===yi||ke===Ae?r.NONE:r.BROWSER_DEFAULT_WEBGL;r.pixelStorei(r.UNPACK_FLIP_Y_WEBGL,w.flipY),r.pixelStorei(r.UNPACK_PREMULTIPLY_ALPHA_WEBGL,w.premultiplyAlpha),r.pixelStorei(r.UNPACK_ALIGNMENT,w.unpackAlignment),r.pixelStorei(r.UNPACK_COLORSPACE_CONVERSION_WEBGL,He);const Oe=w.isCompressedTexture||w.image[0].isCompressedTexture,me=w.image[0]&&w.image[0].isDataTexture,be=[];for(let B=0;B<6;B++)!Oe&&!me?be[B]=P(w.image[B],!0,a.maxCubemapSize):be[B]=me?w.image[B].image:w.image[B],be[B]=Ne(w,be[B]);const je=be[0],We=o.convert(w.format,w.colorSpace),De=o.convert(w.type),qe=L(w.internalFormat,We,De,w.colorSpace),k=w.isVideoTexture!==!0,Re=ne.__version===void 0||re===!0,Te=he.dataReady;let Se=z(w,je);ot(r.TEXTURE_CUBE_MAP,w);let H;if(Oe){k&&Re&&t.texStorage2D(r.TEXTURE_CUBE_MAP,Se,qe,je.width,je.height);for(let B=0;B<6;B++){H=be[B].mipmaps;for(let Le=0;Le<H.length;Le++){const Ke=H[Le];w.format!==Nn?We!==null?k?Te&&t.compressedTexSubImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le,0,0,Ke.width,Ke.height,We,Ke.data):t.compressedTexImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le,qe,Ke.width,Ke.height,0,Ke.data):Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):k?Te&&t.texSubImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le,0,0,Ke.width,Ke.height,We,De,Ke.data):t.texImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le,qe,Ke.width,Ke.height,0,We,De,Ke.data)}}}else{if(H=w.mipmaps,k&&Re){H.length>0&&Se++;const B=$e(be[0]);t.texStorage2D(r.TEXTURE_CUBE_MAP,Se,qe,B.width,B.height)}for(let B=0;B<6;B++)if(me){k?Te&&t.texSubImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,0,0,0,be[B].width,be[B].height,We,De,be[B].data):t.texImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,0,qe,be[B].width,be[B].height,0,We,De,be[B].data);for(let Le=0;Le<H.length;Le++){const vt=H[Le].image[B].image;k?Te&&t.texSubImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le+1,0,0,vt.width,vt.height,We,De,vt.data):t.texImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le+1,qe,vt.width,vt.height,0,We,De,vt.data)}}else{k?Te&&t.texSubImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,0,0,0,We,De,be[B]):t.texImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,0,qe,We,De,be[B]);for(let Le=0;Le<H.length;Le++){const Ke=H[Le];k?Te&&t.texSubImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le+1,0,0,We,De,Ke.image[B]):t.texImage2D(r.TEXTURE_CUBE_MAP_POSITIVE_X+B,Le+1,qe,We,De,Ke.image[B])}}}E(w)&&m(r.TEXTURE_CUBE_MAP),ne.__version=he.version,w.onUpdate&&w.onUpdate(w)}F.__version=w.version}function ie(F,w,$,re,he,ne){const ke=o.convert($.format,$.colorSpace),Ae=o.convert($.type),He=L($.internalFormat,ke,Ae,$.colorSpace),Oe=n.get(w),me=n.get($);if(me.__renderTarget=w,!Oe.__hasExternalTextures){const be=Math.max(1,w.width>>ne),je=Math.max(1,w.height>>ne);he===r.TEXTURE_3D||he===r.TEXTURE_2D_ARRAY?t.texImage3D(he,ne,He,be,je,w.depth,0,ke,Ae,null):t.texImage2D(he,ne,He,be,je,0,ke,Ae,null)}t.bindFramebuffer(r.FRAMEBUFFER,F),Ie(w)?f.framebufferTexture2DMultisampleEXT(r.FRAMEBUFFER,re,he,me.__webglTexture,0,Dt(w)):(he===r.TEXTURE_2D||he>=r.TEXTURE_CUBE_MAP_POSITIVE_X&&he<=r.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&r.framebufferTexture2D(r.FRAMEBUFFER,re,he,me.__webglTexture,ne),t.bindFramebuffer(r.FRAMEBUFFER,null)}function ye(F,w,$){if(r.bindRenderbuffer(r.RENDERBUFFER,F),w.depthBuffer){const re=w.depthTexture,he=re&&re.isDepthTexture?re.type:null,ne=N(w.stencilBuffer,he),ke=w.stencilBuffer?r.DEPTH_STENCIL_ATTACHMENT:r.DEPTH_ATTACHMENT,Ae=Dt(w);Ie(w)?f.renderbufferStorageMultisampleEXT(r.RENDERBUFFER,Ae,ne,w.width,w.height):$?r.renderbufferStorageMultisample(r.RENDERBUFFER,Ae,ne,w.width,w.height):r.renderbufferStorage(r.RENDERBUFFER,ne,w.width,w.height),r.framebufferRenderbuffer(r.FRAMEBUFFER,ke,r.RENDERBUFFER,F)}else{const re=w.textures;for(let he=0;he<re.length;he++){const ne=re[he],ke=o.convert(ne.format,ne.colorSpace),Ae=o.convert(ne.type),He=L(ne.internalFormat,ke,Ae,ne.colorSpace),Oe=Dt(w);$&&Ie(w)===!1?r.renderbufferStorageMultisample(r.RENDERBUFFER,Oe,He,w.width,w.height):Ie(w)?f.renderbufferStorageMultisampleEXT(r.RENDERBUFFER,Oe,He,w.width,w.height):r.renderbufferStorage(r.RENDERBUFFER,He,w.width,w.height)}}r.bindRenderbuffer(r.RENDERBUFFER,null)}function Be(F,w){if(w&&w.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(r.FRAMEBUFFER,F),!(w.depthTexture&&w.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const re=n.get(w.depthTexture);re.__renderTarget=w,(!re.__webglTexture||w.depthTexture.image.width!==w.width||w.depthTexture.image.height!==w.height)&&(w.depthTexture.image.width=w.width,w.depthTexture.image.height=w.height,w.depthTexture.needsUpdate=!0),fe(w.depthTexture,0);const he=re.__webglTexture,ne=Dt(w);if(w.depthTexture.format===qr)Ie(w)?f.framebufferTexture2DMultisampleEXT(r.FRAMEBUFFER,r.DEPTH_ATTACHMENT,r.TEXTURE_2D,he,0,ne):r.framebufferTexture2D(r.FRAMEBUFFER,r.DEPTH_ATTACHMENT,r.TEXTURE_2D,he,0);else if(w.depthTexture.format===Kr)Ie(w)?f.framebufferTexture2DMultisampleEXT(r.FRAMEBUFFER,r.DEPTH_STENCIL_ATTACHMENT,r.TEXTURE_2D,he,0,ne):r.framebufferTexture2D(r.FRAMEBUFFER,r.DEPTH_STENCIL_ATTACHMENT,r.TEXTURE_2D,he,0);else throw new Error("Unknown depthTexture format")}function nt(F){const w=n.get(F),$=F.isWebGLCubeRenderTarget===!0;if(w.__boundDepthTexture!==F.depthTexture){const re=F.depthTexture;if(w.__depthDisposeCallback&&w.__depthDisposeCallback(),re){const he=()=>{delete w.__boundDepthTexture,delete w.__depthDisposeCallback,re.removeEventListener("dispose",he)};re.addEventListener("dispose",he),w.__depthDisposeCallback=he}w.__boundDepthTexture=re}if(F.depthTexture&&!w.__autoAllocateDepthBuffer){if($)throw new Error("target.depthTexture not supported in Cube render targets");const re=F.texture.mipmaps;re&&re.length>0?Be(w.__webglFramebuffer[0],F):Be(w.__webglFramebuffer,F)}else if($){w.__webglDepthbuffer=[];for(let re=0;re<6;re++)if(t.bindFramebuffer(r.FRAMEBUFFER,w.__webglFramebuffer[re]),w.__webglDepthbuffer[re]===void 0)w.__webglDepthbuffer[re]=r.createRenderbuffer(),ye(w.__webglDepthbuffer[re],F,!1);else{const he=F.stencilBuffer?r.DEPTH_STENCIL_ATTACHMENT:r.DEPTH_ATTACHMENT,ne=w.__webglDepthbuffer[re];r.bindRenderbuffer(r.RENDERBUFFER,ne),r.framebufferRenderbuffer(r.FRAMEBUFFER,he,r.RENDERBUFFER,ne)}}else{const re=F.texture.mipmaps;if(re&&re.length>0?t.bindFramebuffer(r.FRAMEBUFFER,w.__webglFramebuffer[0]):t.bindFramebuffer(r.FRAMEBUFFER,w.__webglFramebuffer),w.__webglDepthbuffer===void 0)w.__webglDepthbuffer=r.createRenderbuffer(),ye(w.__webglDepthbuffer,F,!1);else{const he=F.stencilBuffer?r.DEPTH_STENCIL_ATTACHMENT:r.DEPTH_ATTACHMENT,ne=w.__webglDepthbuffer;r.bindRenderbuffer(r.RENDERBUFFER,ne),r.framebufferRenderbuffer(r.FRAMEBUFFER,he,r.RENDERBUFFER,ne)}}t.bindFramebuffer(r.FRAMEBUFFER,null)}function Pt(F,w,$){const re=n.get(F);w!==void 0&&ie(re.__webglFramebuffer,F,F.texture,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,0),$!==void 0&&nt(F)}function it(F){const w=F.texture,$=n.get(F),re=n.get(w);F.addEventListener("dispose",V);const he=F.textures,ne=F.isWebGLCubeRenderTarget===!0,ke=he.length>1;if(ke||(re.__webglTexture===void 0&&(re.__webglTexture=r.createTexture()),re.__version=w.version,l.memory.textures++),ne){$.__webglFramebuffer=[];for(let Ae=0;Ae<6;Ae++)if(w.mipmaps&&w.mipmaps.length>0){$.__webglFramebuffer[Ae]=[];for(let He=0;He<w.mipmaps.length;He++)$.__webglFramebuffer[Ae][He]=r.createFramebuffer()}else $.__webglFramebuffer[Ae]=r.createFramebuffer()}else{if(w.mipmaps&&w.mipmaps.length>0){$.__webglFramebuffer=[];for(let Ae=0;Ae<w.mipmaps.length;Ae++)$.__webglFramebuffer[Ae]=r.createFramebuffer()}else $.__webglFramebuffer=r.createFramebuffer();if(ke)for(let Ae=0,He=he.length;Ae<He;Ae++){const Oe=n.get(he[Ae]);Oe.__webglTexture===void 0&&(Oe.__webglTexture=r.createTexture(),l.memory.textures++)}if(F.samples>0&&Ie(F)===!1){$.__webglMultisampledFramebuffer=r.createFramebuffer(),$.__webglColorRenderbuffer=[],t.bindFramebuffer(r.FRAMEBUFFER,$.__webglMultisampledFramebuffer);for(let Ae=0;Ae<he.length;Ae++){const He=he[Ae];$.__webglColorRenderbuffer[Ae]=r.createRenderbuffer(),r.bindRenderbuffer(r.RENDERBUFFER,$.__webglColorRenderbuffer[Ae]);const Oe=o.convert(He.format,He.colorSpace),me=o.convert(He.type),be=L(He.internalFormat,Oe,me,He.colorSpace,F.isXRRenderTarget===!0),je=Dt(F);r.renderbufferStorageMultisample(r.RENDERBUFFER,je,be,F.width,F.height),r.framebufferRenderbuffer(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0+Ae,r.RENDERBUFFER,$.__webglColorRenderbuffer[Ae])}r.bindRenderbuffer(r.RENDERBUFFER,null),F.depthBuffer&&($.__webglDepthRenderbuffer=r.createRenderbuffer(),ye($.__webglDepthRenderbuffer,F,!0)),t.bindFramebuffer(r.FRAMEBUFFER,null)}}if(ne){t.bindTexture(r.TEXTURE_CUBE_MAP,re.__webglTexture),ot(r.TEXTURE_CUBE_MAP,w);for(let Ae=0;Ae<6;Ae++)if(w.mipmaps&&w.mipmaps.length>0)for(let He=0;He<w.mipmaps.length;He++)ie($.__webglFramebuffer[Ae][He],F,w,r.COLOR_ATTACHMENT0,r.TEXTURE_CUBE_MAP_POSITIVE_X+Ae,He);else ie($.__webglFramebuffer[Ae],F,w,r.COLOR_ATTACHMENT0,r.TEXTURE_CUBE_MAP_POSITIVE_X+Ae,0);E(w)&&m(r.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(ke){for(let Ae=0,He=he.length;Ae<He;Ae++){const Oe=he[Ae],me=n.get(Oe);let be=r.TEXTURE_2D;(F.isWebGL3DRenderTarget||F.isWebGLArrayRenderTarget)&&(be=F.isWebGL3DRenderTarget?r.TEXTURE_3D:r.TEXTURE_2D_ARRAY),t.bindTexture(be,me.__webglTexture),ot(be,Oe),ie($.__webglFramebuffer,F,Oe,r.COLOR_ATTACHMENT0+Ae,be,0),E(Oe)&&m(be)}t.unbindTexture()}else{let Ae=r.TEXTURE_2D;if((F.isWebGL3DRenderTarget||F.isWebGLArrayRenderTarget)&&(Ae=F.isWebGL3DRenderTarget?r.TEXTURE_3D:r.TEXTURE_2D_ARRAY),t.bindTexture(Ae,re.__webglTexture),ot(Ae,w),w.mipmaps&&w.mipmaps.length>0)for(let He=0;He<w.mipmaps.length;He++)ie($.__webglFramebuffer[He],F,w,r.COLOR_ATTACHMENT0,Ae,He);else ie($.__webglFramebuffer,F,w,r.COLOR_ATTACHMENT0,Ae,0);E(w)&&m(Ae),t.unbindTexture()}F.depthBuffer&&nt(F)}function Mt(F){const w=F.textures;for(let $=0,re=w.length;$<re;$++){const he=w[$];if(E(he)){const ne=U(F),ke=n.get(he).__webglTexture;t.bindTexture(ne,ke),m(ne),t.unbindTexture()}}}const O=[],ct=[];function lt(F){if(F.samples>0){if(Ie(F)===!1){const w=F.textures,$=F.width,re=F.height;let he=r.COLOR_BUFFER_BIT;const ne=F.stencilBuffer?r.DEPTH_STENCIL_ATTACHMENT:r.DEPTH_ATTACHMENT,ke=n.get(F),Ae=w.length>1;if(Ae)for(let Oe=0;Oe<w.length;Oe++)t.bindFramebuffer(r.FRAMEBUFFER,ke.__webglMultisampledFramebuffer),r.framebufferRenderbuffer(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0+Oe,r.RENDERBUFFER,null),t.bindFramebuffer(r.FRAMEBUFFER,ke.__webglFramebuffer),r.framebufferTexture2D(r.DRAW_FRAMEBUFFER,r.COLOR_ATTACHMENT0+Oe,r.TEXTURE_2D,null,0);t.bindFramebuffer(r.READ_FRAMEBUFFER,ke.__webglMultisampledFramebuffer);const He=F.texture.mipmaps;He&&He.length>0?t.bindFramebuffer(r.DRAW_FRAMEBUFFER,ke.__webglFramebuffer[0]):t.bindFramebuffer(r.DRAW_FRAMEBUFFER,ke.__webglFramebuffer);for(let Oe=0;Oe<w.length;Oe++){if(F.resolveDepthBuffer&&(F.depthBuffer&&(he|=r.DEPTH_BUFFER_BIT),F.stencilBuffer&&F.resolveStencilBuffer&&(he|=r.STENCIL_BUFFER_BIT)),Ae){r.framebufferRenderbuffer(r.READ_FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.RENDERBUFFER,ke.__webglColorRenderbuffer[Oe]);const me=n.get(w[Oe]).__webglTexture;r.framebufferTexture2D(r.DRAW_FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,me,0)}r.blitFramebuffer(0,0,$,re,0,0,$,re,he,r.NEAREST),p===!0&&(O.length=0,ct.length=0,O.push(r.COLOR_ATTACHMENT0+Oe),F.depthBuffer&&F.resolveDepthBuffer===!1&&(O.push(ne),ct.push(ne),r.invalidateFramebuffer(r.DRAW_FRAMEBUFFER,ct)),r.invalidateFramebuffer(r.READ_FRAMEBUFFER,O))}if(t.bindFramebuffer(r.READ_FRAMEBUFFER,null),t.bindFramebuffer(r.DRAW_FRAMEBUFFER,null),Ae)for(let Oe=0;Oe<w.length;Oe++){t.bindFramebuffer(r.FRAMEBUFFER,ke.__webglMultisampledFramebuffer),r.framebufferRenderbuffer(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0+Oe,r.RENDERBUFFER,ke.__webglColorRenderbuffer[Oe]);const me=n.get(w[Oe]).__webglTexture;t.bindFramebuffer(r.FRAMEBUFFER,ke.__webglFramebuffer),r.framebufferTexture2D(r.DRAW_FRAMEBUFFER,r.COLOR_ATTACHMENT0+Oe,r.TEXTURE_2D,me,0)}t.bindFramebuffer(r.DRAW_FRAMEBUFFER,ke.__webglMultisampledFramebuffer)}else if(F.depthBuffer&&F.resolveDepthBuffer===!1&&p){const w=F.stencilBuffer?r.DEPTH_STENCIL_ATTACHMENT:r.DEPTH_ATTACHMENT;r.invalidateFramebuffer(r.DRAW_FRAMEBUFFER,[w])}}}function Dt(F){return Math.min(a.maxSamples,F.samples)}function Ie(F){const w=n.get(F);return F.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&w.__useRenderToTexture!==!1}function Rt(F){const w=l.render.frame;_.get(F)!==w&&(_.set(F,w),F.update())}function Ne(F,w){const $=F.colorSpace,re=F.format,he=F.type;return F.isCompressedTexture===!0||F.isVideoTexture===!0||$!==Tr&&$!==yi&&(yt.getTransfer($)===Ft?(re!==Nn||he!==$n)&&Ze("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Gt("WebGLTextures: Unsupported texture color space:",$)),w}function $e(F){return typeof HTMLImageElement<"u"&&F instanceof HTMLImageElement?(h.width=F.naturalWidth||F.width,h.height=F.naturalHeight||F.height):typeof VideoFrame<"u"&&F instanceof VideoFrame?(h.width=F.displayWidth,h.height=F.displayHeight):(h.width=F.width,h.height=F.height),h}this.allocateTextureUnit=J,this.resetTextureUnits=Y,this.setTexture2D=fe,this.setTexture2DArray=le,this.setTexture3D=Ee,this.setTextureCube=se,this.rebindTextures=Pt,this.setupRenderTarget=it,this.updateRenderTargetMipmap=Mt,this.updateMultisampleRenderTarget=lt,this.setupDepthRenderbuffer=nt,this.setupFrameBufferTexture=ie,this.useMultisampledRTT=Ie}function hS(r,e){function t(n,a=yi){let o;const l=yt.getTransfer(a);if(n===$n)return r.UNSIGNED_BYTE;if(n===oc)return r.UNSIGNED_SHORT_4_4_4_4;if(n===cc)return r.UNSIGNED_SHORT_5_5_5_1;if(n===wu)return r.UNSIGNED_INT_5_9_9_9_REV;if(n===Au)return r.UNSIGNED_INT_10F_11F_11F_REV;if(n===Mu)return r.BYTE;if(n===Tu)return r.SHORT;if(n===jr)return r.UNSIGNED_SHORT;if(n===ac)return r.INT;if(n===Wi)return r.UNSIGNED_INT;if(n===ii)return r.FLOAT;if(n===Rr)return r.HALF_FLOAT;if(n===Ru)return r.ALPHA;if(n===Cu)return r.RGB;if(n===Nn)return r.RGBA;if(n===qr)return r.DEPTH_COMPONENT;if(n===Kr)return r.DEPTH_STENCIL;if(n===Pu)return r.RED;if(n===lc)return r.RED_INTEGER;if(n===uc)return r.RG;if(n===fc)return r.RG_INTEGER;if(n===dc)return r.RGBA_INTEGER;if(n===Ks||n===Zs||n===Js||n===Qs)if(l===Ft)if(o=e.get("WEBGL_compressed_texture_s3tc_srgb"),o!==null){if(n===Ks)return o.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===Zs)return o.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===Js)return o.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===Qs)return o.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(o=e.get("WEBGL_compressed_texture_s3tc"),o!==null){if(n===Ks)return o.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===Zs)return o.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===Js)return o.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===Qs)return o.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Ao||n===Ro||n===Co||n===Po)if(o=e.get("WEBGL_compressed_texture_pvrtc"),o!==null){if(n===Ao)return o.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===Ro)return o.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Co)return o.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===Po)return o.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===Do||n===Lo||n===Fo)if(o=e.get("WEBGL_compressed_texture_etc"),o!==null){if(n===Do||n===Lo)return l===Ft?o.COMPRESSED_SRGB8_ETC2:o.COMPRESSED_RGB8_ETC2;if(n===Fo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:o.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===Io||n===Uo||n===No||n===Oo||n===ko||n===Bo||n===zo||n===Vo||n===Go||n===Ho||n===Wo||n===Xo||n===$o||n===jo)if(o=e.get("WEBGL_compressed_texture_astc"),o!==null){if(n===Io)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:o.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Uo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:o.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===No)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:o.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Oo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:o.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===ko)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:o.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Bo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:o.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===zo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:o.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Vo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:o.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Go)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:o.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Ho)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:o.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Wo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:o.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Xo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:o.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===$o)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:o.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===jo)return l===Ft?o.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:o.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Yo||n===qo||n===Ko)if(o=e.get("EXT_texture_compression_bptc"),o!==null){if(n===Yo)return l===Ft?o.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:o.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===qo)return o.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Ko)return o.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Zo||n===Jo||n===Qo||n===ec)if(o=e.get("EXT_texture_compression_rgtc"),o!==null){if(n===Zo)return o.COMPRESSED_RED_RGTC1_EXT;if(n===Jo)return o.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===Qo)return o.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===ec)return o.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===Yr?r.UNSIGNED_INT_24_8:r[n]!==void 0?r[n]:null}return{convert:t}}const pS=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,mS=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class _S{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new Xu(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new ui({vertexShader:pS,fragmentShader:mS,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new li(new is(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class xS extends ji{constructor(e,t){super();const n=this;let a=null,o=1,l=null,f="local-floor",p=1,h=null,_=null,g=null,v=null,b=null,R=null;const P=typeof XRWebGLBinding<"u",E=new _S,m={},U=t.getContextAttributes();let L=null,N=null;const z=[],A=[],V=new et;let Z=null;const D=new Tn;D.viewport=new zt;const y=new Tn;y.viewport=new zt;const G=[D,y],Y=new U_;let J=null,ee=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(ce){let de=z[ce];return de===void 0&&(de=new no,z[ce]=de),de.getTargetRaySpace()},this.getControllerGrip=function(ce){let de=z[ce];return de===void 0&&(de=new no,z[ce]=de),de.getGripSpace()},this.getHand=function(ce){let de=z[ce];return de===void 0&&(de=new no,z[ce]=de),de.getHandSpace()};function fe(ce){const de=A.indexOf(ce.inputSource);if(de===-1)return;const ie=z[de];ie!==void 0&&(ie.update(ce.inputSource,ce.frame,h||l),ie.dispatchEvent({type:ce.type,data:ce.inputSource}))}function le(){a.removeEventListener("select",fe),a.removeEventListener("selectstart",fe),a.removeEventListener("selectend",fe),a.removeEventListener("squeeze",fe),a.removeEventListener("squeezestart",fe),a.removeEventListener("squeezeend",fe),a.removeEventListener("end",le),a.removeEventListener("inputsourceschange",Ee);for(let ce=0;ce<z.length;ce++){const de=A[ce];de!==null&&(A[ce]=null,z[ce].disconnect(de))}J=null,ee=null,E.reset();for(const ce in m)delete m[ce];e.setRenderTarget(L),b=null,v=null,g=null,a=null,N=null,at.stop(),n.isPresenting=!1,e.setPixelRatio(Z),e.setSize(V.width,V.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(ce){o=ce,n.isPresenting===!0&&Ze("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(ce){f=ce,n.isPresenting===!0&&Ze("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return h||l},this.setReferenceSpace=function(ce){h=ce},this.getBaseLayer=function(){return v!==null?v:b},this.getBinding=function(){return g===null&&P&&(g=new XRWebGLBinding(a,t)),g},this.getFrame=function(){return R},this.getSession=function(){return a},this.setSession=async function(ce){if(a=ce,a!==null){if(L=e.getRenderTarget(),a.addEventListener("select",fe),a.addEventListener("selectstart",fe),a.addEventListener("selectend",fe),a.addEventListener("squeeze",fe),a.addEventListener("squeezestart",fe),a.addEventListener("squeezeend",fe),a.addEventListener("end",le),a.addEventListener("inputsourceschange",Ee),U.xrCompatible!==!0&&await t.makeXRCompatible(),Z=e.getPixelRatio(),e.getSize(V),P&&"createProjectionLayer"in XRWebGLBinding.prototype){let ie=null,ye=null,Be=null;U.depth&&(Be=U.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,ie=U.stencil?Kr:qr,ye=U.stencil?Yr:Wi);const nt={colorFormat:t.RGBA8,depthFormat:Be,scaleFactor:o};g=this.getBinding(),v=g.createProjectionLayer(nt),a.updateRenderState({layers:[v]}),e.setPixelRatio(1),e.setSize(v.textureWidth,v.textureHeight,!1),N=new $i(v.textureWidth,v.textureHeight,{format:Nn,type:$n,depthTexture:new Wu(v.textureWidth,v.textureHeight,ye,void 0,void 0,void 0,void 0,void 0,void 0,ie),stencilBuffer:U.stencil,colorSpace:e.outputColorSpace,samples:U.antialias?4:0,resolveDepthBuffer:v.ignoreDepthValues===!1,resolveStencilBuffer:v.ignoreDepthValues===!1})}else{const ie={antialias:U.antialias,alpha:!0,depth:U.depth,stencil:U.stencil,framebufferScaleFactor:o};b=new XRWebGLLayer(a,t,ie),a.updateRenderState({baseLayer:b}),e.setPixelRatio(1),e.setSize(b.framebufferWidth,b.framebufferHeight,!1),N=new $i(b.framebufferWidth,b.framebufferHeight,{format:Nn,type:$n,colorSpace:e.outputColorSpace,stencilBuffer:U.stencil,resolveDepthBuffer:b.ignoreDepthValues===!1,resolveStencilBuffer:b.ignoreDepthValues===!1})}N.isXRRenderTarget=!0,this.setFoveation(p),h=null,l=await a.requestReferenceSpace(f),at.setContext(a),at.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(a!==null)return a.environmentBlendMode},this.getDepthTexture=function(){return E.getDepthTexture()};function Ee(ce){for(let de=0;de<ce.removed.length;de++){const ie=ce.removed[de],ye=A.indexOf(ie);ye>=0&&(A[ye]=null,z[ye].disconnect(ie))}for(let de=0;de<ce.added.length;de++){const ie=ce.added[de];let ye=A.indexOf(ie);if(ye===-1){for(let nt=0;nt<z.length;nt++)if(nt>=A.length){A.push(ie),ye=nt;break}else if(A[nt]===null){A[nt]=ie,ye=nt;break}if(ye===-1)break}const Be=z[ye];Be&&Be.connect(ie)}}const se=new K,Me=new K;function we(ce,de,ie){se.setFromMatrixPosition(de.matrixWorld),Me.setFromMatrixPosition(ie.matrixWorld);const ye=se.distanceTo(Me),Be=de.projectionMatrix.elements,nt=ie.projectionMatrix.elements,Pt=Be[14]/(Be[10]-1),it=Be[14]/(Be[10]+1),Mt=(Be[9]+1)/Be[5],O=(Be[9]-1)/Be[5],ct=(Be[8]-1)/Be[0],lt=(nt[8]+1)/nt[0],Dt=Pt*ct,Ie=Pt*lt,Rt=ye/(-ct+lt),Ne=Rt*-ct;if(de.matrixWorld.decompose(ce.position,ce.quaternion,ce.scale),ce.translateX(Ne),ce.translateZ(Rt),ce.matrixWorld.compose(ce.position,ce.quaternion,ce.scale),ce.matrixWorldInverse.copy(ce.matrixWorld).invert(),Be[10]===-1)ce.projectionMatrix.copy(de.projectionMatrix),ce.projectionMatrixInverse.copy(de.projectionMatrixInverse);else{const $e=Pt+Rt,F=it+Rt,w=Dt-Ne,$=Ie+(ye-Ne),re=Mt*it/F*$e,he=O*it/F*$e;ce.projectionMatrix.makePerspective(w,$,re,he,$e,F),ce.projectionMatrixInverse.copy(ce.projectionMatrix).invert()}}function Ge(ce,de){de===null?ce.matrixWorld.copy(ce.matrix):ce.matrixWorld.multiplyMatrices(de.matrixWorld,ce.matrix),ce.matrixWorldInverse.copy(ce.matrixWorld).invert()}this.updateCamera=function(ce){if(a===null)return;let de=ce.near,ie=ce.far;E.texture!==null&&(E.depthNear>0&&(de=E.depthNear),E.depthFar>0&&(ie=E.depthFar)),Y.near=y.near=D.near=de,Y.far=y.far=D.far=ie,(J!==Y.near||ee!==Y.far)&&(a.updateRenderState({depthNear:Y.near,depthFar:Y.far}),J=Y.near,ee=Y.far),Y.layers.mask=ce.layers.mask|6,D.layers.mask=Y.layers.mask&3,y.layers.mask=Y.layers.mask&5;const ye=ce.parent,Be=Y.cameras;Ge(Y,ye);for(let nt=0;nt<Be.length;nt++)Ge(Be[nt],ye);Be.length===2?we(Y,D,y):Y.projectionMatrix.copy(D.projectionMatrix),ot(ce,Y,ye)};function ot(ce,de,ie){ie===null?ce.matrix.copy(de.matrixWorld):(ce.matrix.copy(ie.matrixWorld),ce.matrix.invert(),ce.matrix.multiply(de.matrixWorld)),ce.matrix.decompose(ce.position,ce.quaternion,ce.scale),ce.updateMatrixWorld(!0),ce.projectionMatrix.copy(de.projectionMatrix),ce.projectionMatrixInverse.copy(de.projectionMatrixInverse),ce.isPerspectiveCamera&&(ce.fov=tc*2*Math.atan(1/ce.projectionMatrix.elements[5]),ce.zoom=1)}this.getCamera=function(){return Y},this.getFoveation=function(){if(!(v===null&&b===null))return p},this.setFoveation=function(ce){p=ce,v!==null&&(v.fixedFoveation=ce),b!==null&&b.fixedFoveation!==void 0&&(b.fixedFoveation=ce)},this.hasDepthSensing=function(){return E.texture!==null},this.getDepthSensingMesh=function(){return E.getMesh(Y)},this.getCameraTexture=function(ce){return m[ce]};let _t=null;function Je(ce,de){if(_=de.getViewerPose(h||l),R=de,_!==null){const ie=_.views;b!==null&&(e.setRenderTargetFramebuffer(N,b.framebuffer),e.setRenderTarget(N));let ye=!1;ie.length!==Y.cameras.length&&(Y.cameras.length=0,ye=!0);for(let it=0;it<ie.length;it++){const Mt=ie[it];let O=null;if(b!==null)O=b.getViewport(Mt);else{const lt=g.getViewSubImage(v,Mt);O=lt.viewport,it===0&&(e.setRenderTargetTextures(N,lt.colorTexture,lt.depthStencilTexture),e.setRenderTarget(N))}let ct=G[it];ct===void 0&&(ct=new Tn,ct.layers.enable(it),ct.viewport=new zt,G[it]=ct),ct.matrix.fromArray(Mt.transform.matrix),ct.matrix.decompose(ct.position,ct.quaternion,ct.scale),ct.projectionMatrix.fromArray(Mt.projectionMatrix),ct.projectionMatrixInverse.copy(ct.projectionMatrix).invert(),ct.viewport.set(O.x,O.y,O.width,O.height),it===0&&(Y.matrix.copy(ct.matrix),Y.matrix.decompose(Y.position,Y.quaternion,Y.scale)),ye===!0&&Y.cameras.push(ct)}const Be=a.enabledFeatures;if(Be&&Be.includes("depth-sensing")&&a.depthUsage=="gpu-optimized"&&P){g=n.getBinding();const it=g.getDepthInformation(ie[0]);it&&it.isValid&&it.texture&&E.init(it,a.renderState)}if(Be&&Be.includes("camera-access")&&P){e.state.unbindTexture(),g=n.getBinding();for(let it=0;it<ie.length;it++){const Mt=ie[it].camera;if(Mt){let O=m[Mt];O||(O=new Xu,m[Mt]=O);const ct=g.getCameraImage(Mt);O.sourceTexture=ct}}}}for(let ie=0;ie<z.length;ie++){const ye=A[ie],Be=z[ie];ye!==null&&Be!==void 0&&Be.update(ye,de,h||l)}_t&&_t(ce,de),de.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:de}),R=null}const at=new Yu;at.setAnimationLoop(Je),this.setAnimationLoop=function(ce){_t=ce},this.dispose=function(){}}}const Bi=new jn,gS=new Bt;function vS(r,e){function t(E,m){E.matrixAutoUpdate===!0&&E.updateMatrix(),m.value.copy(E.matrix)}function n(E,m){m.color.getRGB(E.fogColor.value,zu(r)),m.isFog?(E.fogNear.value=m.near,E.fogFar.value=m.far):m.isFogExp2&&(E.fogDensity.value=m.density)}function a(E,m,U,L,N){m.isMeshBasicMaterial||m.isMeshLambertMaterial?o(E,m):m.isMeshToonMaterial?(o(E,m),g(E,m)):m.isMeshPhongMaterial?(o(E,m),_(E,m)):m.isMeshStandardMaterial?(o(E,m),v(E,m),m.isMeshPhysicalMaterial&&b(E,m,N)):m.isMeshMatcapMaterial?(o(E,m),R(E,m)):m.isMeshDepthMaterial?o(E,m):m.isMeshDistanceMaterial?(o(E,m),P(E,m)):m.isMeshNormalMaterial?o(E,m):m.isLineBasicMaterial?(l(E,m),m.isLineDashedMaterial&&f(E,m)):m.isPointsMaterial?p(E,m,U,L):m.isSpriteMaterial?h(E,m):m.isShadowMaterial?(E.color.value.copy(m.color),E.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function o(E,m){E.opacity.value=m.opacity,m.color&&E.diffuse.value.copy(m.color),m.emissive&&E.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(E.map.value=m.map,t(m.map,E.mapTransform)),m.alphaMap&&(E.alphaMap.value=m.alphaMap,t(m.alphaMap,E.alphaMapTransform)),m.bumpMap&&(E.bumpMap.value=m.bumpMap,t(m.bumpMap,E.bumpMapTransform),E.bumpScale.value=m.bumpScale,m.side===_n&&(E.bumpScale.value*=-1)),m.normalMap&&(E.normalMap.value=m.normalMap,t(m.normalMap,E.normalMapTransform),E.normalScale.value.copy(m.normalScale),m.side===_n&&E.normalScale.value.negate()),m.displacementMap&&(E.displacementMap.value=m.displacementMap,t(m.displacementMap,E.displacementMapTransform),E.displacementScale.value=m.displacementScale,E.displacementBias.value=m.displacementBias),m.emissiveMap&&(E.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,E.emissiveMapTransform)),m.specularMap&&(E.specularMap.value=m.specularMap,t(m.specularMap,E.specularMapTransform)),m.alphaTest>0&&(E.alphaTest.value=m.alphaTest);const U=e.get(m),L=U.envMap,N=U.envMapRotation;L&&(E.envMap.value=L,Bi.copy(N),Bi.x*=-1,Bi.y*=-1,Bi.z*=-1,L.isCubeTexture&&L.isRenderTargetTexture===!1&&(Bi.y*=-1,Bi.z*=-1),E.envMapRotation.value.setFromMatrix4(gS.makeRotationFromEuler(Bi)),E.flipEnvMap.value=L.isCubeTexture&&L.isRenderTargetTexture===!1?-1:1,E.reflectivity.value=m.reflectivity,E.ior.value=m.ior,E.refractionRatio.value=m.refractionRatio),m.lightMap&&(E.lightMap.value=m.lightMap,E.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,E.lightMapTransform)),m.aoMap&&(E.aoMap.value=m.aoMap,E.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,E.aoMapTransform))}function l(E,m){E.diffuse.value.copy(m.color),E.opacity.value=m.opacity,m.map&&(E.map.value=m.map,t(m.map,E.mapTransform))}function f(E,m){E.dashSize.value=m.dashSize,E.totalSize.value=m.dashSize+m.gapSize,E.scale.value=m.scale}function p(E,m,U,L){E.diffuse.value.copy(m.color),E.opacity.value=m.opacity,E.size.value=m.size*U,E.scale.value=L*.5,m.map&&(E.map.value=m.map,t(m.map,E.uvTransform)),m.alphaMap&&(E.alphaMap.value=m.alphaMap,t(m.alphaMap,E.alphaMapTransform)),m.alphaTest>0&&(E.alphaTest.value=m.alphaTest)}function h(E,m){E.diffuse.value.copy(m.color),E.opacity.value=m.opacity,E.rotation.value=m.rotation,m.map&&(E.map.value=m.map,t(m.map,E.mapTransform)),m.alphaMap&&(E.alphaMap.value=m.alphaMap,t(m.alphaMap,E.alphaMapTransform)),m.alphaTest>0&&(E.alphaTest.value=m.alphaTest)}function _(E,m){E.specular.value.copy(m.specular),E.shininess.value=Math.max(m.shininess,1e-4)}function g(E,m){m.gradientMap&&(E.gradientMap.value=m.gradientMap)}function v(E,m){E.metalness.value=m.metalness,m.metalnessMap&&(E.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,E.metalnessMapTransform)),E.roughness.value=m.roughness,m.roughnessMap&&(E.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,E.roughnessMapTransform)),m.envMap&&(E.envMapIntensity.value=m.envMapIntensity)}function b(E,m,U){E.ior.value=m.ior,m.sheen>0&&(E.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),E.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(E.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,E.sheenColorMapTransform)),m.sheenRoughnessMap&&(E.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,E.sheenRoughnessMapTransform))),m.clearcoat>0&&(E.clearcoat.value=m.clearcoat,E.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(E.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,E.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(E.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,E.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(E.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,E.clearcoatNormalMapTransform),E.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===_n&&E.clearcoatNormalScale.value.negate())),m.dispersion>0&&(E.dispersion.value=m.dispersion),m.iridescence>0&&(E.iridescence.value=m.iridescence,E.iridescenceIOR.value=m.iridescenceIOR,E.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],E.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(E.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,E.iridescenceMapTransform)),m.iridescenceThicknessMap&&(E.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,E.iridescenceThicknessMapTransform))),m.transmission>0&&(E.transmission.value=m.transmission,E.transmissionSamplerMap.value=U.texture,E.transmissionSamplerSize.value.set(U.width,U.height),m.transmissionMap&&(E.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,E.transmissionMapTransform)),E.thickness.value=m.thickness,m.thicknessMap&&(E.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,E.thicknessMapTransform)),E.attenuationDistance.value=m.attenuationDistance,E.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(E.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(E.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,E.anisotropyMapTransform))),E.specularIntensity.value=m.specularIntensity,E.specularColor.value.copy(m.specularColor),m.specularColorMap&&(E.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,E.specularColorMapTransform)),m.specularIntensityMap&&(E.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,E.specularIntensityMapTransform))}function R(E,m){m.matcap&&(E.matcap.value=m.matcap)}function P(E,m){const U=e.get(m).light;E.referencePosition.value.setFromMatrixPosition(U.matrixWorld),E.nearDistance.value=U.shadow.camera.near,E.farDistance.value=U.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:a}}function yS(r,e,t,n){let a={},o={},l=[];const f=r.getParameter(r.MAX_UNIFORM_BUFFER_BINDINGS);function p(U,L){const N=L.program;n.uniformBlockBinding(U,N)}function h(U,L){let N=a[U.id];N===void 0&&(R(U),N=_(U),a[U.id]=N,U.addEventListener("dispose",E));const z=L.program;n.updateUBOMapping(U,z);const A=e.render.frame;o[U.id]!==A&&(v(U),o[U.id]=A)}function _(U){const L=g();U.__bindingPointIndex=L;const N=r.createBuffer(),z=U.__size,A=U.usage;return r.bindBuffer(r.UNIFORM_BUFFER,N),r.bufferData(r.UNIFORM_BUFFER,z,A),r.bindBuffer(r.UNIFORM_BUFFER,null),r.bindBufferBase(r.UNIFORM_BUFFER,L,N),N}function g(){for(let U=0;U<f;U++)if(l.indexOf(U)===-1)return l.push(U),U;return Gt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function v(U){const L=a[U.id],N=U.uniforms,z=U.__cache;r.bindBuffer(r.UNIFORM_BUFFER,L);for(let A=0,V=N.length;A<V;A++){const Z=Array.isArray(N[A])?N[A]:[N[A]];for(let D=0,y=Z.length;D<y;D++){const G=Z[D];if(b(G,A,D,z)===!0){const Y=G.__offset,J=Array.isArray(G.value)?G.value:[G.value];let ee=0;for(let fe=0;fe<J.length;fe++){const le=J[fe],Ee=P(le);typeof le=="number"||typeof le=="boolean"?(G.__data[0]=le,r.bufferSubData(r.UNIFORM_BUFFER,Y+ee,G.__data)):le.isMatrix3?(G.__data[0]=le.elements[0],G.__data[1]=le.elements[1],G.__data[2]=le.elements[2],G.__data[3]=0,G.__data[4]=le.elements[3],G.__data[5]=le.elements[4],G.__data[6]=le.elements[5],G.__data[7]=0,G.__data[8]=le.elements[6],G.__data[9]=le.elements[7],G.__data[10]=le.elements[8],G.__data[11]=0):(le.toArray(G.__data,ee),ee+=Ee.storage/Float32Array.BYTES_PER_ELEMENT)}r.bufferSubData(r.UNIFORM_BUFFER,Y,G.__data)}}}r.bindBuffer(r.UNIFORM_BUFFER,null)}function b(U,L,N,z){const A=U.value,V=L+"_"+N;if(z[V]===void 0)return typeof A=="number"||typeof A=="boolean"?z[V]=A:z[V]=A.clone(),!0;{const Z=z[V];if(typeof A=="number"||typeof A=="boolean"){if(Z!==A)return z[V]=A,!0}else if(Z.equals(A)===!1)return Z.copy(A),!0}return!1}function R(U){const L=U.uniforms;let N=0;const z=16;for(let V=0,Z=L.length;V<Z;V++){const D=Array.isArray(L[V])?L[V]:[L[V]];for(let y=0,G=D.length;y<G;y++){const Y=D[y],J=Array.isArray(Y.value)?Y.value:[Y.value];for(let ee=0,fe=J.length;ee<fe;ee++){const le=J[ee],Ee=P(le),se=N%z,Me=se%Ee.boundary,we=se+Me;N+=Me,we!==0&&z-we<Ee.storage&&(N+=z-we),Y.__data=new Float32Array(Ee.storage/Float32Array.BYTES_PER_ELEMENT),Y.__offset=N,N+=Ee.storage}}}const A=N%z;return A>0&&(N+=z-A),U.__size=N,U.__cache={},this}function P(U){const L={boundary:0,storage:0};return typeof U=="number"||typeof U=="boolean"?(L.boundary=4,L.storage=4):U.isVector2?(L.boundary=8,L.storage=8):U.isVector3||U.isColor?(L.boundary=16,L.storage=12):U.isVector4?(L.boundary=16,L.storage=16):U.isMatrix3?(L.boundary=48,L.storage=48):U.isMatrix4?(L.boundary=64,L.storage=64):U.isTexture?Ze("WebGLRenderer: Texture samplers can not be part of an uniforms group."):Ze("WebGLRenderer: Unsupported uniform value type.",U),L}function E(U){const L=U.target;L.removeEventListener("dispose",E);const N=l.indexOf(L.__bindingPointIndex);l.splice(N,1),r.deleteBuffer(a[L.id]),delete a[L.id],delete o[L.id]}function m(){for(const U in a)r.deleteBuffer(a[U]);l=[],a={},o={}}return{bind:p,update:h,dispose:m}}const SS=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let ei=null;function bS(){return ei===null&&(ei=new E_(SS,32,32,uc,Rr),ei.minFilter=wn,ei.magFilter=wn,ei.wrapS=ni,ei.wrapT=ni,ei.generateMipmaps=!1,ei.needsUpdate=!0),ei}class ES{constructor(e={}){const{canvas:t=Y0(),context:n=null,depth:a=!0,stencil:o=!1,alpha:l=!1,antialias:f=!1,premultipliedAlpha:p=!0,preserveDrawingBuffer:h=!1,powerPreference:_="default",failIfMajorPerformanceCaveat:g=!1,reversedDepthBuffer:v=!1}=e;this.isWebGLRenderer=!0;let b;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");b=n.getContextAttributes().alpha}else b=l;const R=new Set([dc,fc,lc]),P=new Set([$n,Wi,jr,Yr,oc,cc]),E=new Uint32Array(4),m=new Int32Array(4);let U=null,L=null;const N=[],z=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=bi,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const A=this;let V=!1;this._outputColorSpace=Mn;let Z=0,D=0,y=null,G=-1,Y=null;const J=new zt,ee=new zt;let fe=null;const le=new xt(0);let Ee=0,se=t.width,Me=t.height,we=1,Ge=null,ot=null;const _t=new zt(0,0,se,Me),Je=new zt(0,0,se,Me);let at=!1;const ce=new mc;let de=!1,ie=!1;const ye=new Bt,Be=new K,nt=new zt,Pt={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let it=!1;function Mt(){return y===null?we:1}let O=n;function ct(C,W){return t.getContext(C,W)}try{const C={alpha:!0,depth:a,stencil:o,antialias:f,premultipliedAlpha:p,preserveDrawingBuffer:h,powerPreference:_,failIfMajorPerformanceCaveat:g};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${sc}`),t.addEventListener("webglcontextlost",H,!1),t.addEventListener("webglcontextrestored",B,!1),t.addEventListener("webglcontextcreationerror",Le,!1),O===null){const W="webgl2";if(O=ct(W,C),O===null)throw ct(W)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(C){throw C("WebGLRenderer: "+C.message),C}let lt,Dt,Ie,Rt,Ne,$e,F,w,$,re,he,ne,ke,Ae,He,Oe,me,be,je,We,De,qe,k,Re;function Te(){lt=new Dv(O),lt.init(),qe=new hS(O,lt),Dt=new bv(O,lt,e,qe),Ie=new fS(O,lt),Dt.reversedDepthBuffer&&v&&Ie.buffers.depth.setReversed(!0),Rt=new Iv(O),Ne=new Jy,$e=new dS(O,lt,Ie,Ne,Dt,qe,Rt),F=new Mv(A),w=new Pv(A),$=new k_(O),k=new yv(O,$),re=new Lv(O,$,Rt,k),he=new Nv(O,re,$,Rt),je=new Uv(O,Dt,$e),Oe=new Ev(Ne),ne=new Zy(A,F,w,lt,Dt,k,Oe),ke=new vS(A,Ne),Ae=new eS,He=new aS(lt),be=new vv(A,F,w,Ie,he,b,p),me=new lS(A,he,Dt),Re=new yS(O,Rt,Dt,Ie),We=new Sv(O,lt,Rt),De=new Fv(O,lt,Rt),Rt.programs=ne.programs,A.capabilities=Dt,A.extensions=lt,A.properties=Ne,A.renderLists=Ae,A.shadowMap=me,A.state=Ie,A.info=Rt}Te();const Se=new xS(A,O);this.xr=Se,this.getContext=function(){return O},this.getContextAttributes=function(){return O.getContextAttributes()},this.forceContextLoss=function(){const C=lt.get("WEBGL_lose_context");C&&C.loseContext()},this.forceContextRestore=function(){const C=lt.get("WEBGL_lose_context");C&&C.restoreContext()},this.getPixelRatio=function(){return we},this.setPixelRatio=function(C){C!==void 0&&(we=C,this.setSize(se,Me,!1))},this.getSize=function(C){return C.set(se,Me)},this.setSize=function(C,W,te=!0){if(Se.isPresenting){Ze("WebGLRenderer: Can't change size while VR device is presenting.");return}se=C,Me=W,t.width=Math.floor(C*we),t.height=Math.floor(W*we),te===!0&&(t.style.width=C+"px",t.style.height=W+"px"),this.setViewport(0,0,C,W)},this.getDrawingBufferSize=function(C){return C.set(se*we,Me*we).floor()},this.setDrawingBufferSize=function(C,W,te){se=C,Me=W,we=te,t.width=Math.floor(C*te),t.height=Math.floor(W*te),this.setViewport(0,0,C,W)},this.getCurrentViewport=function(C){return C.copy(J)},this.getViewport=function(C){return C.copy(_t)},this.setViewport=function(C,W,te,Q){C.isVector4?_t.set(C.x,C.y,C.z,C.w):_t.set(C,W,te,Q),Ie.viewport(J.copy(_t).multiplyScalar(we).round())},this.getScissor=function(C){return C.copy(Je)},this.setScissor=function(C,W,te,Q){C.isVector4?Je.set(C.x,C.y,C.z,C.w):Je.set(C,W,te,Q),Ie.scissor(ee.copy(Je).multiplyScalar(we).round())},this.getScissorTest=function(){return at},this.setScissorTest=function(C){Ie.setScissorTest(at=C)},this.setOpaqueSort=function(C){Ge=C},this.setTransparentSort=function(C){ot=C},this.getClearColor=function(C){return C.copy(be.getClearColor())},this.setClearColor=function(){be.setClearColor(...arguments)},this.getClearAlpha=function(){return be.getClearAlpha()},this.setClearAlpha=function(){be.setClearAlpha(...arguments)},this.clear=function(C=!0,W=!0,te=!0){let Q=0;if(C){let j=!1;if(y!==null){const ve=y.texture.format;j=R.has(ve)}if(j){const ve=y.texture.type,Pe=P.has(ve),Fe=be.getClearColor(),Ue=be.getClearAlpha(),ge=Fe.r,Ye=Fe.g,ze=Fe.b;Pe?(E[0]=ge,E[1]=Ye,E[2]=ze,E[3]=Ue,O.clearBufferuiv(O.COLOR,0,E)):(m[0]=ge,m[1]=Ye,m[2]=ze,m[3]=Ue,O.clearBufferiv(O.COLOR,0,m))}else Q|=O.COLOR_BUFFER_BIT}W&&(Q|=O.DEPTH_BUFFER_BIT),te&&(Q|=O.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),O.clear(Q)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",H,!1),t.removeEventListener("webglcontextrestored",B,!1),t.removeEventListener("webglcontextcreationerror",Le,!1),be.dispose(),Ae.dispose(),He.dispose(),Ne.dispose(),F.dispose(),w.dispose(),he.dispose(),k.dispose(),Re.dispose(),ne.dispose(),Se.dispose(),Se.removeEventListener("sessionstart",ss),Se.removeEventListener("sessionend",as),Jt.stop()};function H(C){C.preventDefault(),Sl("WebGLRenderer: Context Lost."),V=!0}function B(){Sl("WebGLRenderer: Context Restored."),V=!1;const C=Rt.autoReset,W=me.enabled,te=me.autoUpdate,Q=me.needsUpdate,j=me.type;Te(),Rt.autoReset=C,me.enabled=W,me.autoUpdate=te,me.needsUpdate=Q,me.type=j}function Le(C){Gt("WebGLRenderer: A WebGL context could not be created. Reason: ",C.statusMessage)}function Ke(C){const W=C.target;W.removeEventListener("dispose",Ke),vt(W)}function vt(C){ut(C),Ne.remove(C)}function ut(C){const W=Ne.get(C).programs;W!==void 0&&(W.forEach(function(te){ne.releaseProgram(te)}),C.isShaderMaterial&&ne.releaseShaderCache(C))}this.renderBufferDirect=function(C,W,te,Q,j,ve){W===null&&(W=Pt);const Pe=j.isMesh&&j.matrixWorld.determinant()<0,Fe=ma(C,W,te,Q,j);Ie.setMaterial(Q,Pe);let Ue=te.index,ge=1;if(Q.wireframe===!0){if(Ue=re.getWireframeAttribute(te),Ue===void 0)return;ge=2}const Ye=te.drawRange,ze=te.attributes.position;let ft=Ye.start*ge,bt=(Ye.start+Ye.count)*ge;ve!==null&&(ft=Math.max(ft,ve.start*ge),bt=Math.min(bt,(ve.start+ve.count)*ge)),Ue!==null?(ft=Math.max(ft,0),bt=Math.min(bt,Ue.count)):ze!=null&&(ft=Math.max(ft,0),bt=Math.min(bt,ze.count));const kt=bt-ft;if(kt<0||kt===1/0)return;k.setup(j,Q,Fe,te,Ue);let Nt,Ct=We;if(Ue!==null&&(Nt=$.get(Ue),Ct=De,Ct.setIndex(Nt)),j.isMesh)Q.wireframe===!0?(Ie.setLineWidth(Q.wireframeLinewidth*Mt()),Ct.setMode(O.LINES)):Ct.setMode(O.TRIANGLES);else if(j.isLine){let Ve=Q.linewidth;Ve===void 0&&(Ve=1),Ie.setLineWidth(Ve*Mt()),j.isLineSegments?Ct.setMode(O.LINES):j.isLineLoop?Ct.setMode(O.LINE_LOOP):Ct.setMode(O.LINE_STRIP)}else j.isPoints?Ct.setMode(O.POINTS):j.isSprite&&Ct.setMode(O.TRIANGLES);if(j.isBatchedMesh)if(j._multiDrawInstances!==null)Zr("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),Ct.renderMultiDrawInstances(j._multiDrawStarts,j._multiDrawCounts,j._multiDrawCount,j._multiDrawInstances);else if(lt.get("WEBGL_multi_draw"))Ct.renderMultiDraw(j._multiDrawStarts,j._multiDrawCounts,j._multiDrawCount);else{const Ve=j._multiDrawStarts,Ot=j._multiDrawCounts,pt=j._multiDrawCount,S=Ue?$.get(Ue).bytesPerElement:1,Et=Ne.get(Q).currentProgram.getUniforms();for(let on=0;on<pt;on++)Et.setValue(O,"_gl_DrawID",on),Ct.render(Ve[on]/S,Ot[on])}else if(j.isInstancedMesh)Ct.renderInstances(ft,kt,j.count);else if(te.isInstancedBufferGeometry){const Ve=te._maxInstanceCount!==void 0?te._maxInstanceCount:1/0,Ot=Math.min(te.instanceCount,Ve);Ct.renderInstances(ft,kt,Ot)}else Ct.render(ft,kt)};function yn(C,W,te){C.transparent===!0&&C.side===Hn&&C.forceSinglePass===!1?(C.side=_n,C.needsUpdate=!0,qi(C,W,te),C.side=Ei,C.needsUpdate=!0,qi(C,W,te),C.side=Hn):qi(C,W,te)}this.compile=function(C,W,te=null){te===null&&(te=C),L=He.get(te),L.init(W),z.push(L),te.traverseVisible(function(j){j.isLight&&j.layers.test(W.layers)&&(L.pushLight(j),j.castShadow&&L.pushShadow(j))}),C!==te&&C.traverseVisible(function(j){j.isLight&&j.layers.test(W.layers)&&(L.pushLight(j),j.castShadow&&L.pushShadow(j))}),L.setupLights();const Q=new Set;return C.traverse(function(j){if(!(j.isMesh||j.isPoints||j.isLine||j.isSprite))return;const ve=j.material;if(ve)if(Array.isArray(ve))for(let Pe=0;Pe<ve.length;Pe++){const Fe=ve[Pe];yn(Fe,te,j),Q.add(Fe)}else yn(ve,te,j),Q.add(ve)}),L=z.pop(),Q},this.compileAsync=function(C,W,te=null){const Q=this.compile(C,W,te);return new Promise(j=>{function ve(){if(Q.forEach(function(Pe){Ne.get(Pe).currentProgram.isReady()&&Q.delete(Pe)}),Q.size===0){j(C);return}setTimeout(ve,10)}lt.get("KHR_parallel_shader_compile")!==null?ve():setTimeout(ve,10)})};let Vt=null;function Yi(C){Vt&&Vt(C)}function ss(){Jt.stop()}function as(){Jt.start()}const Jt=new Yu;Jt.setAnimationLoop(Yi),typeof self<"u"&&Jt.setContext(self),this.setAnimationLoop=function(C){Vt=C,Se.setAnimationLoop(C),C===null?Jt.stop():Jt.start()},Se.addEventListener("sessionstart",ss),Se.addEventListener("sessionend",as),this.render=function(C,W){if(W!==void 0&&W.isCamera!==!0){Gt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(V===!0)return;if(C.matrixWorldAutoUpdate===!0&&C.updateMatrixWorld(),W.parent===null&&W.matrixWorldAutoUpdate===!0&&W.updateMatrixWorld(),Se.enabled===!0&&Se.isPresenting===!0&&(Se.cameraAutoUpdate===!0&&Se.updateCamera(W),W=Se.getCamera()),C.isScene===!0&&C.onBeforeRender(A,C,W,y),L=He.get(C,z.length),L.init(W),z.push(L),ye.multiplyMatrices(W.projectionMatrix,W.matrixWorldInverse),ce.setFromProjectionMatrix(ye,Wn,W.reversedDepth),ie=this.localClippingEnabled,de=Oe.init(this.clippingPlanes,ie),U=Ae.get(C,N.length),U.init(),N.push(U),Se.enabled===!0&&Se.isPresenting===!0){const ve=A.xr.getDepthSensingMesh();ve!==null&&Lr(ve,W,-1/0,A.sortObjects)}Lr(C,W,0,A.sortObjects),U.finish(),A.sortObjects===!0&&U.sort(Ge,ot),it=Se.enabled===!1||Se.isPresenting===!1||Se.hasDepthSensing()===!1,it&&be.addToRenderList(U,C),this.info.render.frame++,de===!0&&Oe.beginShadows();const te=L.state.shadowsArray;me.render(te,C,W),de===!0&&Oe.endShadows(),this.info.autoReset===!0&&this.info.reset();const Q=U.opaque,j=U.transmissive;if(L.setupLights(),W.isArrayCamera){const ve=W.cameras;if(j.length>0)for(let Pe=0,Fe=ve.length;Pe<Fe;Pe++){const Ue=ve[Pe];Ai(Q,j,C,Ue)}it&&be.render(C);for(let Pe=0,Fe=ve.length;Pe<Fe;Pe++){const Ue=ve[Pe];wi(U,C,Ue,Ue.viewport)}}else j.length>0&&Ai(Q,j,C,W),it&&be.render(C),wi(U,C,W);y!==null&&D===0&&($e.updateMultisampleRenderTarget(y),$e.updateRenderTargetMipmap(y)),C.isScene===!0&&C.onAfterRender(A,C,W),k.resetDefaultState(),G=-1,Y=null,z.pop(),z.length>0?(L=z[z.length-1],de===!0&&Oe.setGlobalState(A.clippingPlanes,L.state.camera)):L=null,N.pop(),N.length>0?U=N[N.length-1]:U=null};function Lr(C,W,te,Q){if(C.visible===!1)return;if(C.layers.test(W.layers)){if(C.isGroup)te=C.renderOrder;else if(C.isLOD)C.autoUpdate===!0&&C.update(W);else if(C.isLight)L.pushLight(C),C.castShadow&&L.pushShadow(C);else if(C.isSprite){if(!C.frustumCulled||ce.intersectsSprite(C)){Q&&nt.setFromMatrixPosition(C.matrixWorld).applyMatrix4(ye);const Pe=he.update(C),Fe=C.material;Fe.visible&&U.push(C,Pe,Fe,te,nt.z,null)}}else if((C.isMesh||C.isLine||C.isPoints)&&(!C.frustumCulled||ce.intersectsObject(C))){const Pe=he.update(C),Fe=C.material;if(Q&&(C.boundingSphere!==void 0?(C.boundingSphere===null&&C.computeBoundingSphere(),nt.copy(C.boundingSphere.center)):(Pe.boundingSphere===null&&Pe.computeBoundingSphere(),nt.copy(Pe.boundingSphere.center)),nt.applyMatrix4(C.matrixWorld).applyMatrix4(ye)),Array.isArray(Fe)){const Ue=Pe.groups;for(let ge=0,Ye=Ue.length;ge<Ye;ge++){const ze=Ue[ge],ft=Fe[ze.materialIndex];ft&&ft.visible&&U.push(C,Pe,ft,te,nt.z,ze)}}else Fe.visible&&U.push(C,Pe,Fe,te,nt.z,null)}}const ve=C.children;for(let Pe=0,Fe=ve.length;Pe<Fe;Pe++)Lr(ve[Pe],W,te,Q)}function wi(C,W,te,Q){const{opaque:j,transmissive:ve,transparent:Pe}=C;L.setupLightsView(te),de===!0&&Oe.setGlobalState(A.clippingPlanes,te),Q&&Ie.viewport(J.copy(Q)),j.length>0&&fi(j,W,te),ve.length>0&&fi(ve,W,te),Pe.length>0&&fi(Pe,W,te),Ie.buffers.depth.setTest(!0),Ie.buffers.depth.setMask(!0),Ie.buffers.color.setMask(!0),Ie.setPolygonOffset(!1)}function Ai(C,W,te,Q){if((te.isScene===!0?te.overrideMaterial:null)!==null)return;L.state.transmissionRenderTarget[Q.id]===void 0&&(L.state.transmissionRenderTarget[Q.id]=new $i(1,1,{generateMipmaps:!0,type:lt.has("EXT_color_buffer_half_float")||lt.has("EXT_color_buffer_float")?Rr:$n,minFilter:Hi,samples:4,stencilBuffer:o,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:yt.workingColorSpace}));const ve=L.state.transmissionRenderTarget[Q.id],Pe=Q.viewport||J;ve.setSize(Pe.z*A.transmissionResolutionScale,Pe.w*A.transmissionResolutionScale);const Fe=A.getRenderTarget(),Ue=A.getActiveCubeFace(),ge=A.getActiveMipmapLevel();A.setRenderTarget(ve),A.getClearColor(le),Ee=A.getClearAlpha(),Ee<1&&A.setClearColor(16777215,.5),A.clear(),it&&be.render(te);const Ye=A.toneMapping;A.toneMapping=bi;const ze=Q.viewport;if(Q.viewport!==void 0&&(Q.viewport=void 0),L.setupLightsView(Q),de===!0&&Oe.setGlobalState(A.clippingPlanes,Q),fi(C,te,Q),$e.updateMultisampleRenderTarget(ve),$e.updateRenderTargetMipmap(ve),lt.has("WEBGL_multisampled_render_to_texture")===!1){let ft=!1;for(let bt=0,kt=W.length;bt<kt;bt++){const Nt=W[bt],{object:Ct,geometry:Ve,material:Ot,group:pt}=Nt;if(Ot.side===Hn&&Ct.layers.test(Q.layers)){const S=Ot.side;Ot.side=_n,Ot.needsUpdate=!0,os(Ct,te,Q,Ve,Ot,pt),Ot.side=S,Ot.needsUpdate=!0,ft=!0}}ft===!0&&($e.updateMultisampleRenderTarget(ve),$e.updateRenderTargetMipmap(ve))}A.setRenderTarget(Fe,Ue,ge),A.setClearColor(le,Ee),ze!==void 0&&(Q.viewport=ze),A.toneMapping=Ye}function fi(C,W,te){const Q=W.isScene===!0?W.overrideMaterial:null;for(let j=0,ve=C.length;j<ve;j++){const Pe=C[j],{object:Fe,geometry:Ue,group:ge}=Pe;let Ye=Pe.material;Ye.allowOverride===!0&&Q!==null&&(Ye=Q),Fe.layers.test(te.layers)&&os(Fe,W,te,Ue,Ye,ge)}}function os(C,W,te,Q,j,ve){C.onBeforeRender(A,W,te,Q,j,ve),C.modelViewMatrix.multiplyMatrices(te.matrixWorldInverse,C.matrixWorld),C.normalMatrix.getNormalMatrix(C.modelViewMatrix),j.onBeforeRender(A,W,te,Q,C,ve),j.transparent===!0&&j.side===Hn&&j.forceSinglePass===!1?(j.side=_n,j.needsUpdate=!0,A.renderBufferDirect(te,W,Q,j,C,ve),j.side=Ei,j.needsUpdate=!0,A.renderBufferDirect(te,W,Q,j,C,ve),j.side=Hn):A.renderBufferDirect(te,W,Q,j,C,ve),C.onAfterRender(A,W,te,Q,j,ve)}function qi(C,W,te){W.isScene!==!0&&(W=Pt);const Q=Ne.get(C),j=L.state.lights,ve=L.state.shadowsArray,Pe=j.state.version,Fe=ne.getParameters(C,j.state,ve,W,te),Ue=ne.getProgramCacheKey(Fe);let ge=Q.programs;Q.environment=C.isMeshStandardMaterial?W.environment:null,Q.fog=W.fog,Q.envMap=(C.isMeshStandardMaterial?w:F).get(C.envMap||Q.environment),Q.envMapRotation=Q.environment!==null&&C.envMap===null?W.environmentRotation:C.envMapRotation,ge===void 0&&(C.addEventListener("dispose",Ke),ge=new Map,Q.programs=ge);let Ye=ge.get(Ue);if(Ye!==void 0){if(Q.currentProgram===Ye&&Q.lightsStateVersion===Pe)return Fr(C,Fe),Ye}else Fe.uniforms=ne.getUniforms(C),C.onBeforeCompile(Fe,A),Ye=ne.acquireProgram(Fe,Ue),ge.set(Ue,Ye),Q.uniforms=Fe.uniforms;const ze=Q.uniforms;return(!C.isShaderMaterial&&!C.isRawShaderMaterial||C.clipping===!0)&&(ze.clippingPlanes=Oe.uniform),Fr(C,Fe),Q.needsLights=xa(C),Q.lightsStateVersion=Pe,Q.needsLights&&(ze.ambientLightColor.value=j.state.ambient,ze.lightProbe.value=j.state.probe,ze.directionalLights.value=j.state.directional,ze.directionalLightShadows.value=j.state.directionalShadow,ze.spotLights.value=j.state.spot,ze.spotLightShadows.value=j.state.spotShadow,ze.rectAreaLights.value=j.state.rectArea,ze.ltc_1.value=j.state.rectAreaLTC1,ze.ltc_2.value=j.state.rectAreaLTC2,ze.pointLights.value=j.state.point,ze.pointLightShadows.value=j.state.pointShadow,ze.hemisphereLights.value=j.state.hemi,ze.directionalShadowMap.value=j.state.directionalShadowMap,ze.directionalShadowMatrix.value=j.state.directionalShadowMatrix,ze.spotShadowMap.value=j.state.spotShadowMap,ze.spotLightMatrix.value=j.state.spotLightMatrix,ze.spotLightMap.value=j.state.spotLightMap,ze.pointShadowMap.value=j.state.pointShadowMap,ze.pointShadowMatrix.value=j.state.pointShadowMatrix),Q.currentProgram=Ye,Q.uniformsList=null,Ye}function cs(C){if(C.uniformsList===null){const W=C.currentProgram.getUniforms();C.uniformsList=ta.seqWithValue(W.seq,C.uniforms)}return C.uniformsList}function Fr(C,W){const te=Ne.get(C);te.outputColorSpace=W.outputColorSpace,te.batching=W.batching,te.batchingColor=W.batchingColor,te.instancing=W.instancing,te.instancingColor=W.instancingColor,te.instancingMorph=W.instancingMorph,te.skinning=W.skinning,te.morphTargets=W.morphTargets,te.morphNormals=W.morphNormals,te.morphColors=W.morphColors,te.morphTargetsCount=W.morphTargetsCount,te.numClippingPlanes=W.numClippingPlanes,te.numIntersection=W.numClipIntersection,te.vertexAlphas=W.vertexAlphas,te.vertexTangents=W.vertexTangents,te.toneMapping=W.toneMapping}function ma(C,W,te,Q,j){W.isScene!==!0&&(W=Pt),$e.resetTextureUnits();const ve=W.fog,Pe=Q.isMeshStandardMaterial?W.environment:null,Fe=y===null?A.outputColorSpace:y.isXRRenderTarget===!0?y.texture.colorSpace:Tr,Ue=(Q.isMeshStandardMaterial?w:F).get(Q.envMap||Pe),ge=Q.vertexColors===!0&&!!te.attributes.color&&te.attributes.color.itemSize===4,Ye=!!te.attributes.tangent&&(!!Q.normalMap||Q.anisotropy>0),ze=!!te.morphAttributes.position,ft=!!te.morphAttributes.normal,bt=!!te.morphAttributes.color;let kt=bi;Q.toneMapped&&(y===null||y.isXRRenderTarget===!0)&&(kt=A.toneMapping);const Nt=te.morphAttributes.position||te.morphAttributes.normal||te.morphAttributes.color,Ct=Nt!==void 0?Nt.length:0,Ve=Ne.get(Q),Ot=L.state.lights;if(de===!0&&(ie===!0||C!==Y)){const Qt=C===Y&&Q.id===G;Oe.setState(Q,C,Qt)}let pt=!1;Q.version===Ve.__version?(Ve.needsLights&&Ve.lightsStateVersion!==Ot.state.version||Ve.outputColorSpace!==Fe||j.isBatchedMesh&&Ve.batching===!1||!j.isBatchedMesh&&Ve.batching===!0||j.isBatchedMesh&&Ve.batchingColor===!0&&j.colorTexture===null||j.isBatchedMesh&&Ve.batchingColor===!1&&j.colorTexture!==null||j.isInstancedMesh&&Ve.instancing===!1||!j.isInstancedMesh&&Ve.instancing===!0||j.isSkinnedMesh&&Ve.skinning===!1||!j.isSkinnedMesh&&Ve.skinning===!0||j.isInstancedMesh&&Ve.instancingColor===!0&&j.instanceColor===null||j.isInstancedMesh&&Ve.instancingColor===!1&&j.instanceColor!==null||j.isInstancedMesh&&Ve.instancingMorph===!0&&j.morphTexture===null||j.isInstancedMesh&&Ve.instancingMorph===!1&&j.morphTexture!==null||Ve.envMap!==Ue||Q.fog===!0&&Ve.fog!==ve||Ve.numClippingPlanes!==void 0&&(Ve.numClippingPlanes!==Oe.numPlanes||Ve.numIntersection!==Oe.numIntersection)||Ve.vertexAlphas!==ge||Ve.vertexTangents!==Ye||Ve.morphTargets!==ze||Ve.morphNormals!==ft||Ve.morphColors!==bt||Ve.toneMapping!==kt||Ve.morphTargetsCount!==Ct)&&(pt=!0):(pt=!0,Ve.__version=Q.version);let S=Ve.currentProgram;pt===!0&&(S=qi(Q,W,j));let Et=!1,on=!1,On=!1;const wt=S.getUniforms(),tn=Ve.uniforms;if(Ie.useProgram(S.program)&&(Et=!0,on=!0,On=!0),Q.id!==G&&(G=Q.id,on=!0),Et||Y!==C){Ie.buffers.depth.getReversed()&&C.reversedDepth!==!0&&(C._reversedDepth=!0,C.updateProjectionMatrix()),wt.setValue(O,"projectionMatrix",C.projectionMatrix),wt.setValue(O,"viewMatrix",C.matrixWorldInverse);const nn=wt.map.cameraPosition;nn!==void 0&&nn.setValue(O,Be.setFromMatrixPosition(C.matrixWorld)),Dt.logarithmicDepthBuffer&&wt.setValue(O,"logDepthBufFC",2/(Math.log(C.far+1)/Math.LN2)),(Q.isMeshPhongMaterial||Q.isMeshToonMaterial||Q.isMeshLambertMaterial||Q.isMeshBasicMaterial||Q.isMeshStandardMaterial||Q.isShaderMaterial)&&wt.setValue(O,"isOrthographic",C.isOrthographicCamera===!0),Y!==C&&(Y=C,on=!0,On=!0)}if(j.isSkinnedMesh){wt.setOptional(O,j,"bindMatrix"),wt.setOptional(O,j,"bindMatrixInverse");const Qt=j.skeleton;Qt&&(Qt.boneTexture===null&&Qt.computeBoneTexture(),wt.setValue(O,"boneTexture",Qt.boneTexture,$e))}j.isBatchedMesh&&(wt.setOptional(O,j,"batchingTexture"),wt.setValue(O,"batchingTexture",j._matricesTexture,$e),wt.setOptional(O,j,"batchingIdTexture"),wt.setValue(O,"batchingIdTexture",j._indirectTexture,$e),wt.setOptional(O,j,"batchingColorTexture"),j._colorsTexture!==null&&wt.setValue(O,"batchingColorTexture",j._colorsTexture,$e));const dn=te.morphAttributes;if((dn.position!==void 0||dn.normal!==void 0||dn.color!==void 0)&&je.update(j,te,S),(on||Ve.receiveShadow!==j.receiveShadow)&&(Ve.receiveShadow=j.receiveShadow,wt.setValue(O,"receiveShadow",j.receiveShadow)),Q.isMeshGouraudMaterial&&Q.envMap!==null&&(tn.envMap.value=Ue,tn.flipEnvMap.value=Ue.isCubeTexture&&Ue.isRenderTargetTexture===!1?-1:1),Q.isMeshStandardMaterial&&Q.envMap===null&&W.environment!==null&&(tn.envMapIntensity.value=W.environmentIntensity),tn.dfgLUT!==void 0&&(tn.dfgLUT.value=bS()),on&&(wt.setValue(O,"toneMappingExposure",A.toneMappingExposure),Ve.needsLights&&_a(tn,On),ve&&Q.fog===!0&&ke.refreshFogUniforms(tn,ve),ke.refreshMaterialUniforms(tn,Q,we,Me,L.state.transmissionRenderTarget[C.id]),ta.upload(O,cs(Ve),tn,$e)),Q.isShaderMaterial&&Q.uniformsNeedUpdate===!0&&(ta.upload(O,cs(Ve),tn,$e),Q.uniformsNeedUpdate=!1),Q.isSpriteMaterial&&wt.setValue(O,"center",j.center),wt.setValue(O,"modelViewMatrix",j.modelViewMatrix),wt.setValue(O,"normalMatrix",j.normalMatrix),wt.setValue(O,"modelMatrix",j.matrixWorld),Q.isShaderMaterial||Q.isRawShaderMaterial){const Qt=Q.uniformsGroups;for(let nn=0,Ir=Qt.length;nn<Ir;nn++){const Yn=Qt[nn];Re.update(Yn,S),Re.bind(Yn,S)}}return S}function _a(C,W){C.ambientLightColor.needsUpdate=W,C.lightProbe.needsUpdate=W,C.directionalLights.needsUpdate=W,C.directionalLightShadows.needsUpdate=W,C.pointLights.needsUpdate=W,C.pointLightShadows.needsUpdate=W,C.spotLights.needsUpdate=W,C.spotLightShadows.needsUpdate=W,C.rectAreaLights.needsUpdate=W,C.hemisphereLights.needsUpdate=W}function xa(C){return C.isMeshLambertMaterial||C.isMeshToonMaterial||C.isMeshPhongMaterial||C.isMeshStandardMaterial||C.isShadowMaterial||C.isShaderMaterial&&C.lights===!0}this.getActiveCubeFace=function(){return Z},this.getActiveMipmapLevel=function(){return D},this.getRenderTarget=function(){return y},this.setRenderTargetTextures=function(C,W,te){const Q=Ne.get(C);Q.__autoAllocateDepthBuffer=C.resolveDepthBuffer===!1,Q.__autoAllocateDepthBuffer===!1&&(Q.__useRenderToTexture=!1),Ne.get(C.texture).__webglTexture=W,Ne.get(C.depthTexture).__webglTexture=Q.__autoAllocateDepthBuffer?void 0:te,Q.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(C,W){const te=Ne.get(C);te.__webglFramebuffer=W,te.__useDefaultFramebuffer=W===void 0};const ga=O.createFramebuffer();this.setRenderTarget=function(C,W=0,te=0){y=C,Z=W,D=te;let Q=!0,j=null,ve=!1,Pe=!1;if(C){const Ue=Ne.get(C);if(Ue.__useDefaultFramebuffer!==void 0)Ie.bindFramebuffer(O.FRAMEBUFFER,null),Q=!1;else if(Ue.__webglFramebuffer===void 0)$e.setupRenderTarget(C);else if(Ue.__hasExternalTextures)$e.rebindTextures(C,Ne.get(C.texture).__webglTexture,Ne.get(C.depthTexture).__webglTexture);else if(C.depthBuffer){const ze=C.depthTexture;if(Ue.__boundDepthTexture!==ze){if(ze!==null&&Ne.has(ze)&&(C.width!==ze.image.width||C.height!==ze.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");$e.setupDepthRenderbuffer(C)}}const ge=C.texture;(ge.isData3DTexture||ge.isDataArrayTexture||ge.isCompressedArrayTexture)&&(Pe=!0);const Ye=Ne.get(C).__webglFramebuffer;C.isWebGLCubeRenderTarget?(Array.isArray(Ye[W])?j=Ye[W][te]:j=Ye[W],ve=!0):C.samples>0&&$e.useMultisampledRTT(C)===!1?j=Ne.get(C).__webglMultisampledFramebuffer:Array.isArray(Ye)?j=Ye[te]:j=Ye,J.copy(C.viewport),ee.copy(C.scissor),fe=C.scissorTest}else J.copy(_t).multiplyScalar(we).floor(),ee.copy(Je).multiplyScalar(we).floor(),fe=at;if(te!==0&&(j=ga),Ie.bindFramebuffer(O.FRAMEBUFFER,j)&&Q&&Ie.drawBuffers(C,j),Ie.viewport(J),Ie.scissor(ee),Ie.setScissorTest(fe),ve){const Ue=Ne.get(C.texture);O.framebufferTexture2D(O.FRAMEBUFFER,O.COLOR_ATTACHMENT0,O.TEXTURE_CUBE_MAP_POSITIVE_X+W,Ue.__webglTexture,te)}else if(Pe){const Ue=W;for(let ge=0;ge<C.textures.length;ge++){const Ye=Ne.get(C.textures[ge]);O.framebufferTextureLayer(O.FRAMEBUFFER,O.COLOR_ATTACHMENT0+ge,Ye.__webglTexture,te,Ue)}}else if(C!==null&&te!==0){const Ue=Ne.get(C.texture);O.framebufferTexture2D(O.FRAMEBUFFER,O.COLOR_ATTACHMENT0,O.TEXTURE_2D,Ue.__webglTexture,te)}G=-1},this.readRenderTargetPixels=function(C,W,te,Q,j,ve,Pe,Fe=0){if(!(C&&C.isWebGLRenderTarget)){Gt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Ue=Ne.get(C).__webglFramebuffer;if(C.isWebGLCubeRenderTarget&&Pe!==void 0&&(Ue=Ue[Pe]),Ue){Ie.bindFramebuffer(O.FRAMEBUFFER,Ue);try{const ge=C.textures[Fe],Ye=ge.format,ze=ge.type;if(!Dt.textureFormatReadable(Ye)){Gt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!Dt.textureTypeReadable(ze)){Gt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}W>=0&&W<=C.width-Q&&te>=0&&te<=C.height-j&&(C.textures.length>1&&O.readBuffer(O.COLOR_ATTACHMENT0+Fe),O.readPixels(W,te,Q,j,qe.convert(Ye),qe.convert(ze),ve))}finally{const ge=y!==null?Ne.get(y).__webglFramebuffer:null;Ie.bindFramebuffer(O.FRAMEBUFFER,ge)}}},this.readRenderTargetPixelsAsync=async function(C,W,te,Q,j,ve,Pe,Fe=0){if(!(C&&C.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Ue=Ne.get(C).__webglFramebuffer;if(C.isWebGLCubeRenderTarget&&Pe!==void 0&&(Ue=Ue[Pe]),Ue)if(W>=0&&W<=C.width-Q&&te>=0&&te<=C.height-j){Ie.bindFramebuffer(O.FRAMEBUFFER,Ue);const ge=C.textures[Fe],Ye=ge.format,ze=ge.type;if(!Dt.textureFormatReadable(Ye))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!Dt.textureTypeReadable(ze))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const ft=O.createBuffer();O.bindBuffer(O.PIXEL_PACK_BUFFER,ft),O.bufferData(O.PIXEL_PACK_BUFFER,ve.byteLength,O.STREAM_READ),C.textures.length>1&&O.readBuffer(O.COLOR_ATTACHMENT0+Fe),O.readPixels(W,te,Q,j,qe.convert(Ye),qe.convert(ze),0);const bt=y!==null?Ne.get(y).__webglFramebuffer:null;Ie.bindFramebuffer(O.FRAMEBUFFER,bt);const kt=O.fenceSync(O.SYNC_GPU_COMMANDS_COMPLETE,0);return O.flush(),await q0(O,kt,4),O.bindBuffer(O.PIXEL_PACK_BUFFER,ft),O.getBufferSubData(O.PIXEL_PACK_BUFFER,0,ve),O.deleteBuffer(ft),O.deleteSync(kt),ve}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(C,W=null,te=0){const Q=Math.pow(2,-te),j=Math.floor(C.image.width*Q),ve=Math.floor(C.image.height*Q),Pe=W!==null?W.x:0,Fe=W!==null?W.y:0;$e.setTexture2D(C,0),O.copyTexSubImage2D(O.TEXTURE_2D,te,0,0,Pe,Fe,j,ve),Ie.unbindTexture()};const Tt=O.createFramebuffer(),va=O.createFramebuffer();this.copyTextureToTexture=function(C,W,te=null,Q=null,j=0,ve=null){ve===null&&(j!==0?(Zr("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),ve=j,j=0):ve=0);let Pe,Fe,Ue,ge,Ye,ze,ft,bt,kt;const Nt=C.isCompressedTexture?C.mipmaps[ve]:C.image;if(te!==null)Pe=te.max.x-te.min.x,Fe=te.max.y-te.min.y,Ue=te.isBox3?te.max.z-te.min.z:1,ge=te.min.x,Ye=te.min.y,ze=te.isBox3?te.min.z:0;else{const dn=Math.pow(2,-j);Pe=Math.floor(Nt.width*dn),Fe=Math.floor(Nt.height*dn),C.isDataArrayTexture?Ue=Nt.depth:C.isData3DTexture?Ue=Math.floor(Nt.depth*dn):Ue=1,ge=0,Ye=0,ze=0}Q!==null?(ft=Q.x,bt=Q.y,kt=Q.z):(ft=0,bt=0,kt=0);const Ct=qe.convert(W.format),Ve=qe.convert(W.type);let Ot;W.isData3DTexture?($e.setTexture3D(W,0),Ot=O.TEXTURE_3D):W.isDataArrayTexture||W.isCompressedArrayTexture?($e.setTexture2DArray(W,0),Ot=O.TEXTURE_2D_ARRAY):($e.setTexture2D(W,0),Ot=O.TEXTURE_2D),O.pixelStorei(O.UNPACK_FLIP_Y_WEBGL,W.flipY),O.pixelStorei(O.UNPACK_PREMULTIPLY_ALPHA_WEBGL,W.premultiplyAlpha),O.pixelStorei(O.UNPACK_ALIGNMENT,W.unpackAlignment);const pt=O.getParameter(O.UNPACK_ROW_LENGTH),S=O.getParameter(O.UNPACK_IMAGE_HEIGHT),Et=O.getParameter(O.UNPACK_SKIP_PIXELS),on=O.getParameter(O.UNPACK_SKIP_ROWS),On=O.getParameter(O.UNPACK_SKIP_IMAGES);O.pixelStorei(O.UNPACK_ROW_LENGTH,Nt.width),O.pixelStorei(O.UNPACK_IMAGE_HEIGHT,Nt.height),O.pixelStorei(O.UNPACK_SKIP_PIXELS,ge),O.pixelStorei(O.UNPACK_SKIP_ROWS,Ye),O.pixelStorei(O.UNPACK_SKIP_IMAGES,ze);const wt=C.isDataArrayTexture||C.isData3DTexture,tn=W.isDataArrayTexture||W.isData3DTexture;if(C.isDepthTexture){const dn=Ne.get(C),Qt=Ne.get(W),nn=Ne.get(dn.__renderTarget),Ir=Ne.get(Qt.__renderTarget);Ie.bindFramebuffer(O.READ_FRAMEBUFFER,nn.__webglFramebuffer),Ie.bindFramebuffer(O.DRAW_FRAMEBUFFER,Ir.__webglFramebuffer);for(let Yn=0;Yn<Ue;Yn++)wt&&(O.framebufferTextureLayer(O.READ_FRAMEBUFFER,O.COLOR_ATTACHMENT0,Ne.get(C).__webglTexture,j,ze+Yn),O.framebufferTextureLayer(O.DRAW_FRAMEBUFFER,O.COLOR_ATTACHMENT0,Ne.get(W).__webglTexture,ve,kt+Yn)),O.blitFramebuffer(ge,Ye,Pe,Fe,ft,bt,Pe,Fe,O.DEPTH_BUFFER_BIT,O.NEAREST);Ie.bindFramebuffer(O.READ_FRAMEBUFFER,null),Ie.bindFramebuffer(O.DRAW_FRAMEBUFFER,null)}else if(j!==0||C.isRenderTargetTexture||Ne.has(C)){const dn=Ne.get(C),Qt=Ne.get(W);Ie.bindFramebuffer(O.READ_FRAMEBUFFER,Tt),Ie.bindFramebuffer(O.DRAW_FRAMEBUFFER,va);for(let nn=0;nn<Ue;nn++)wt?O.framebufferTextureLayer(O.READ_FRAMEBUFFER,O.COLOR_ATTACHMENT0,dn.__webglTexture,j,ze+nn):O.framebufferTexture2D(O.READ_FRAMEBUFFER,O.COLOR_ATTACHMENT0,O.TEXTURE_2D,dn.__webglTexture,j),tn?O.framebufferTextureLayer(O.DRAW_FRAMEBUFFER,O.COLOR_ATTACHMENT0,Qt.__webglTexture,ve,kt+nn):O.framebufferTexture2D(O.DRAW_FRAMEBUFFER,O.COLOR_ATTACHMENT0,O.TEXTURE_2D,Qt.__webglTexture,ve),j!==0?O.blitFramebuffer(ge,Ye,Pe,Fe,ft,bt,Pe,Fe,O.COLOR_BUFFER_BIT,O.NEAREST):tn?O.copyTexSubImage3D(Ot,ve,ft,bt,kt+nn,ge,Ye,Pe,Fe):O.copyTexSubImage2D(Ot,ve,ft,bt,ge,Ye,Pe,Fe);Ie.bindFramebuffer(O.READ_FRAMEBUFFER,null),Ie.bindFramebuffer(O.DRAW_FRAMEBUFFER,null)}else tn?C.isDataTexture||C.isData3DTexture?O.texSubImage3D(Ot,ve,ft,bt,kt,Pe,Fe,Ue,Ct,Ve,Nt.data):W.isCompressedArrayTexture?O.compressedTexSubImage3D(Ot,ve,ft,bt,kt,Pe,Fe,Ue,Ct,Nt.data):O.texSubImage3D(Ot,ve,ft,bt,kt,Pe,Fe,Ue,Ct,Ve,Nt):C.isDataTexture?O.texSubImage2D(O.TEXTURE_2D,ve,ft,bt,Pe,Fe,Ct,Ve,Nt.data):C.isCompressedTexture?O.compressedTexSubImage2D(O.TEXTURE_2D,ve,ft,bt,Nt.width,Nt.height,Ct,Nt.data):O.texSubImage2D(O.TEXTURE_2D,ve,ft,bt,Pe,Fe,Ct,Ve,Nt);O.pixelStorei(O.UNPACK_ROW_LENGTH,pt),O.pixelStorei(O.UNPACK_IMAGE_HEIGHT,S),O.pixelStorei(O.UNPACK_SKIP_PIXELS,Et),O.pixelStorei(O.UNPACK_SKIP_ROWS,on),O.pixelStorei(O.UNPACK_SKIP_IMAGES,On),ve===0&&W.generateMipmaps&&O.generateMipmap(Ot),Ie.unbindTexture()},this.initRenderTarget=function(C){Ne.get(C).__webglFramebuffer===void 0&&$e.setupRenderTarget(C)},this.initTexture=function(C){C.isCubeTexture?$e.setTextureCube(C,0):C.isData3DTexture?$e.setTexture3D(C,0):C.isDataArrayTexture||C.isCompressedArrayTexture?$e.setTexture2DArray(C,0):$e.setTexture2D(C,0),Ie.unbindTexture()},this.resetState=function(){Z=0,D=0,y=null,Ie.reset(),k.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Wn}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=yt._getDrawingBufferColorSpace(e),t.unpackColorSpace=yt._getUnpackColorSpace()}}const mu={type:"change"},vc={type:"start"},Qu={type:"end"},Ys=new pc,_u=new vi,MS=Math.cos(70*Z0.DEG2RAD),jt=new K,pn=2*Math.PI,It={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},fo=1e-6;class TS extends N_{constructor(e,t=null){super(e,t),this.state=It.NONE,this.target=new K,this.cursor=new K,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:xr.ROTATE,MIDDLE:xr.DOLLY,RIGHT:xr.PAN},this.touches={ONE:fr.ROTATE,TWO:fr.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this._lastPosition=new K,this._lastQuaternion=new Xi,this._lastTargetPosition=new K,this._quat=new Xi().setFromUnitVectors(e.up,new K(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new Wl,this._sphericalDelta=new Wl,this._scale=1,this._panOffset=new K,this._rotateStart=new et,this._rotateEnd=new et,this._rotateDelta=new et,this._panStart=new et,this._panEnd=new et,this._panDelta=new et,this._dollyStart=new et,this._dollyEnd=new et,this._dollyDelta=new et,this._dollyDirection=new K,this._mouse=new et,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=AS.bind(this),this._onPointerDown=wS.bind(this),this._onPointerUp=RS.bind(this),this._onContextMenu=US.bind(this),this._onMouseWheel=DS.bind(this),this._onKeyDown=LS.bind(this),this._onTouchStart=FS.bind(this),this._onTouchMove=IS.bind(this),this._onMouseDown=CS.bind(this),this._onMouseMove=PS.bind(this),this._interceptControlDown=NS.bind(this),this._interceptControlUp=OS.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(mu),this.update(),this.state=It.NONE}update(e=null){const t=this.object.position;jt.copy(t).sub(this.target),jt.applyQuaternion(this._quat),this._spherical.setFromVector3(jt),this.autoRotate&&this.state===It.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,a=this.maxAzimuthAngle;isFinite(n)&&isFinite(a)&&(n<-Math.PI?n+=pn:n>Math.PI&&(n-=pn),a<-Math.PI?a+=pn:a>Math.PI&&(a-=pn),n<=a?this._spherical.theta=Math.max(n,Math.min(a,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+a)/2?Math.max(n,this._spherical.theta):Math.min(a,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let o=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{const l=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),o=l!=this._spherical.radius}if(jt.setFromSpherical(this._spherical),jt.applyQuaternion(this._quatInverse),t.copy(this.target).add(jt),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let l=null;if(this.object.isPerspectiveCamera){const f=jt.length();l=this._clampDistance(f*this._scale);const p=f-l;this.object.position.addScaledVector(this._dollyDirection,p),this.object.updateMatrixWorld(),o=!!p}else if(this.object.isOrthographicCamera){const f=new K(this._mouse.x,this._mouse.y,0);f.unproject(this.object);const p=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),o=p!==this.object.zoom;const h=new K(this._mouse.x,this._mouse.y,0);h.unproject(this.object),this.object.position.sub(h).add(f),this.object.updateMatrixWorld(),l=jt.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;l!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(l).add(this.object.position):(Ys.origin.copy(this.object.position),Ys.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Ys.direction))<MS?this.object.lookAt(this.target):(_u.setFromNormalAndCoplanarPoint(this.object.up,this.target),Ys.intersectPlane(_u,this.target))))}else if(this.object.isOrthographicCamera){const l=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),l!==this.object.zoom&&(this.object.updateProjectionMatrix(),o=!0)}return this._scale=1,this._performCursorZoom=!1,o||this._lastPosition.distanceToSquared(this.object.position)>fo||8*(1-this._lastQuaternion.dot(this.object.quaternion))>fo||this._lastTargetPosition.distanceToSquared(this.target)>fo?(this.dispatchEvent(mu),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?pn/60*this.autoRotateSpeed*e:pn/60/60*this.autoRotateSpeed}_getZoomScale(e){const t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){jt.setFromMatrixColumn(t,0),jt.multiplyScalar(-e),this._panOffset.add(jt)}_panUp(e,t){this.screenSpacePanning===!0?jt.setFromMatrixColumn(t,1):(jt.setFromMatrixColumn(t,0),jt.crossVectors(this.object.up,jt)),jt.multiplyScalar(e),this._panOffset.add(jt)}_pan(e,t){const n=this.domElement;if(this.object.isPerspectiveCamera){const a=this.object.position;jt.copy(a).sub(this.target);let o=jt.length();o*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*o/n.clientHeight,this.object.matrix),this._panUp(2*t*o/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;const n=this.domElement.getBoundingClientRect(),a=e-n.left,o=t-n.top,l=n.width,f=n.height;this._mouse.x=a/l*2-1,this._mouse.y=-(o/f)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(pn*this._rotateDelta.x/t.clientHeight),this._rotateUp(pn*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(pn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-pn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(pn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-pn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),a=.5*(e.pageY+t.y);this._rotateStart.set(n,a)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),a=.5*(e.pageY+t.y);this._panStart.set(n,a)}}_handleTouchStartDolly(e){const t=this._getSecondPointerPosition(e),n=e.pageX-t.x,a=e.pageY-t.y,o=Math.sqrt(n*n+a*a);this._dollyStart.set(0,o)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{const n=this._getSecondPointerPosition(e),a=.5*(e.pageX+n.x),o=.5*(e.pageY+n.y);this._rotateEnd.set(a,o)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(pn*this._rotateDelta.x/t.clientHeight),this._rotateUp(pn*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),a=.5*(e.pageY+t.y);this._panEnd.set(n,a)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){const t=this._getSecondPointerPosition(e),n=e.pageX-t.x,a=e.pageY-t.y,o=Math.sqrt(n*n+a*a);this._dollyEnd.set(0,o),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);const l=(e.pageX+t.x)*.5,f=(e.pageY+t.y)*.5;this._updateZoomParameters(l,f)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new et,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){const t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){const t=e.deltaMode,n={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}}function wS(r){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(r.pointerId),this.domElement.addEventListener("pointermove",this._onPointerMove),this.domElement.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(r)&&(this._addPointer(r),r.pointerType==="touch"?this._onTouchStart(r):this._onMouseDown(r)))}function AS(r){this.enabled!==!1&&(r.pointerType==="touch"?this._onTouchMove(r):this._onMouseMove(r))}function RS(r){switch(this._removePointer(r),this._pointers.length){case 0:this.domElement.releasePointerCapture(r.pointerId),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Qu),this.state=It.NONE;break;case 1:const e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function CS(r){let e;switch(r.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case xr.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(r),this.state=It.DOLLY;break;case xr.ROTATE:if(r.ctrlKey||r.metaKey||r.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(r),this.state=It.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(r),this.state=It.ROTATE}break;case xr.PAN:if(r.ctrlKey||r.metaKey||r.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(r),this.state=It.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(r),this.state=It.PAN}break;default:this.state=It.NONE}this.state!==It.NONE&&this.dispatchEvent(vc)}function PS(r){switch(this.state){case It.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(r);break;case It.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(r);break;case It.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(r);break}}function DS(r){this.enabled===!1||this.enableZoom===!1||this.state!==It.NONE||(r.preventDefault(),this.dispatchEvent(vc),this._handleMouseWheel(this._customWheelEvent(r)),this.dispatchEvent(Qu))}function LS(r){this.enabled!==!1&&this._handleKeyDown(r)}function FS(r){switch(this._trackPointer(r),this._pointers.length){case 1:switch(this.touches.ONE){case fr.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(r),this.state=It.TOUCH_ROTATE;break;case fr.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(r),this.state=It.TOUCH_PAN;break;default:this.state=It.NONE}break;case 2:switch(this.touches.TWO){case fr.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(r),this.state=It.TOUCH_DOLLY_PAN;break;case fr.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(r),this.state=It.TOUCH_DOLLY_ROTATE;break;default:this.state=It.NONE}break;default:this.state=It.NONE}this.state!==It.NONE&&this.dispatchEvent(vc)}function IS(r){switch(this._trackPointer(r),this.state){case It.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(r),this.update();break;case It.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(r),this.update();break;case It.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(r),this.update();break;case It.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(r),this.update();break;default:this.state=It.NONE}}function US(r){this.enabled!==!1&&r.preventDefault()}function NS(r){r.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function OS(r){r.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}const kS="modulepreload",BS=function(r,e){return new URL(r,e).href},xu={},zS=function(e,t,n){let a=Promise.resolve();if(t&&t.length>0){let h=function(_){return Promise.all(_.map(g=>Promise.resolve(g).then(v=>({status:"fulfilled",value:v}),v=>({status:"rejected",reason:v}))))};const l=document.getElementsByTagName("link"),f=document.querySelector("meta[property=csp-nonce]"),p=f?.nonce||f?.getAttribute("nonce");a=h(t.map(_=>{if(_=BS(_,n),_ in xu)return;xu[_]=!0;const g=_.endsWith(".css"),v=g?'[rel="stylesheet"]':"";if(n)for(let R=l.length-1;R>=0;R--){const P=l[R];if(P.href===_&&(!g||P.rel==="stylesheet"))return}else if(document.querySelector(`link[href="${_}"]${v}`))return;const b=document.createElement("link");if(b.rel=g?"stylesheet":kS,g||(b.as="script"),b.crossOrigin="",b.href=_,p&&b.setAttribute("nonce",p),document.head.appendChild(b),g)return new Promise((R,P)=>{b.addEventListener("load",R),b.addEventListener("error",()=>P(new Error(`Unable to preload CSS for ${_}`)))})}))}function o(l){const f=new Event("vite:preloadError",{cancelable:!0});if(f.payload=l,window.dispatchEvent(f),!f.defaultPrevented)throw l}return a.then(l=>{for(const f of l||[])f.status==="rejected"&&o(f.reason);return e().catch(o)})};var VS=(async function(r={}){var e,t=r,n=typeof window=="object",a=typeof WorkerGlobalScope<"u",o=typeof process=="object"&&process.versions?.node&&process.type!="renderer",l=!n&&!o&&!a;if(o){const{createRequire:i}=await zS(async()=>{const{createRequire:s}=await Promise.resolve().then(()=>KS);return{createRequire:s}},void 0,import.meta.url);var f=i(import.meta.url)}var p="./this.program",h=(i,s)=>{throw s},_=import.meta.url,g="";function v(i){return t.locateFile?t.locateFile(i,g):g+i}var b,R;if(o){if(!(typeof process=="object"&&process.versions?.node&&process.type!="renderer"))throw new Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");var P=process.versions.node,E=P.split(".").slice(0,3);if(E=E[0]*1e4+E[1]*100+E[2].split("-")[0]*1,E<16e4)throw new Error("This emscripten-generated code requires node v16.0.0 (detected v"+P+")");var m=f("fs");_.startsWith("file:")&&(g=f("path").dirname(f("url").fileURLToPath(_))+"/"),R=s=>{s=V(s)?new URL(s):s;var c=m.readFileSync(s);return A(Buffer.isBuffer(c)),c},b=async(s,c=!0)=>{s=V(s)?new URL(s):s;var u=m.readFileSync(s,c?void 0:"utf8");return A(c?Buffer.isBuffer(u):typeof u=="string"),u},process.argv.length>1&&(p=process.argv[1].replace(/\\/g,"/")),process.argv.slice(2),h=(s,c)=>{throw process.exitCode=s,c}}else if(l){if(typeof process=="object"&&process.versions?.node&&process.type!="renderer"||typeof window=="object"||typeof WorkerGlobalScope<"u")throw new Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)")}else if(n||a){try{g=new URL(".",_).href}catch{}if(!(typeof window=="object"||typeof WorkerGlobalScope<"u"))throw new Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");a&&(R=i=>{var s=new XMLHttpRequest;return s.open("GET",i,!1),s.responseType="arraybuffer",s.send(null),new Uint8Array(s.response)}),b=async i=>{if(V(i))return new Promise((c,u)=>{var d=new XMLHttpRequest;d.open("GET",i,!0),d.responseType="arraybuffer",d.onload=()=>{if(d.status==200||d.status==0&&d.response){c(d.response);return}u(d.status)},d.onerror=u,d.send(null)});var s=await fetch(i,{credentials:"same-origin"});if(s.ok)return s.arrayBuffer();throw new Error(s.status+" : "+s.url)}}else throw new Error("environment detection error");var U=console.log.bind(console),L=console.error.bind(console);A(!l,"shell environment detected but not enabled at build time.  Add `shell` to `-sENVIRONMENT` to enable.");var N;typeof WebAssembly!="object"&&L("no native wasm support detected");var z=!1;function A(i,s){i||$("Assertion failed"+(s?": "+s:""))}var V=i=>i.startsWith("file://");function Z(){var i=Da();A((i&3)==0),i==0&&(i+=4),ye[i>>2]=34821223,ye[i+4>>2]=2310721022,ye[0]=1668509029}function D(){if(!z){var i=Da();i==0&&(i+=4);var s=ye[i>>2],c=ye[i+4>>2];(s!=34821223||c!=2310721022)&&$(`Stack overflow! Stack cookie has been overwritten at ${Se(i)}, expected hex dwords 0x89BACDFE and 0x2135467, but received ${Se(c)} ${Se(s)}`),ye[0]!=1668509029&&$("Runtime error: The application has corrupted its heap memory area (address zero)!")}}class y extends Error{}class G extends y{}class Y extends y{constructor(s){super(s),this.excPtr=s;const c=Jc(s);this.name=c[0],this.message=c[1]}}(()=>{var i=new Int16Array(1),s=new Int8Array(i.buffer);if(i[0]=25459,s[0]!==115||s[1]!==99)throw"Runtime error: expected the system to be little-endian! (Run with -sSUPPORT_BIG_ENDIAN to bypass)"})();function J(i){Object.getOwnPropertyDescriptor(t,i)||Object.defineProperty(t,i,{configurable:!0,set(){$(`Attempt to set \`Module.${i}\` after it has already been processed.  This can happen, for example, when code is injected via '--post-js' rather than '--pre-js'`)}})}function ee(i){return()=>A(!1,`call to '${i}' via reference taken before Wasm module initialization`)}function fe(i){Object.getOwnPropertyDescriptor(t,i)&&$(`\`Module.${i}\` was supplied but \`${i}\` not included in INCOMING_MODULE_JS_API`)}function le(i){return i==="FS_createPath"||i==="FS_createDataFile"||i==="FS_createPreloadedFile"||i==="FS_unlink"||i==="addRunDependency"||i==="FS_createLazyFile"||i==="FS_createDevice"||i==="removeRunDependency"}function Ee(i,s){typeof globalThis<"u"&&!Object.getOwnPropertyDescriptor(globalThis,i)&&Object.defineProperty(globalThis,i,{configurable:!0,get(){s()}})}function se(i,s){Ee(i,()=>{Le(`\`${i}\` is not longer defined by emscripten. ${s}`)})}se("buffer","Please use HEAP8.buffer or wasmMemory.buffer"),se("asm","Please use wasmExports instead");function Me(i){Ee(i,()=>{var s=`\`${i}\` is a library symbol and not included by default; add it to your library.js __deps or to DEFAULT_LIBRARY_FUNCS_TO_INCLUDE on the command line`,c=i;c.startsWith("_")||(c="$"+i),s+=` (e.g. -sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE='${c}')`,le(i)&&(s+=". Alternatively, forcing filesystem support (-sFORCE_FILESYSTEM) can export this for you"),Le(s)}),we(i)}function we(i){Object.getOwnPropertyDescriptor(t,i)||Object.defineProperty(t,i,{configurable:!0,get(){var s=`'${i}' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the Emscripten FAQ)`;le(i)&&(s+=". Alternatively, forcing filesystem support (-sFORCE_FILESYSTEM) can export this for you"),$(s)}})}var Ge,ot,_t,Je,at,ce,de,ie,ye,Be,nt,Pt,it,Mt=!1;function O(){var i=_t.buffer;Je=new Int8Array(i),ce=new Int16Array(i),at=new Uint8Array(i),de=new Uint16Array(i),ie=new Int32Array(i),ye=new Uint32Array(i),Be=new Float32Array(i),nt=new Float64Array(i),Pt=new BigInt64Array(i),it=new BigUint64Array(i)}A(typeof Int32Array<"u"&&typeof Float64Array<"u"&&Int32Array.prototype.subarray!=null&&Int32Array.prototype.set!=null,"JS engine does not provide full typed array support");function ct(){if(t.preRun)for(typeof t.preRun=="function"&&(t.preRun=[t.preRun]);t.preRun.length;)Re(t.preRun.shift());J("preRun"),We(k)}function lt(){A(!Mt),Mt=!0,D(),!t.noFSInit&&!S.initialized&&S.init(),Li.__wasm_call_ctors(),S.ignorePermissions=!1}function Dt(){if(D(),t.postRun)for(typeof t.postRun=="function"&&(t.postRun=[t.postRun]);t.postRun.length;)qe(t.postRun.shift());J("postRun"),We(De)}var Ie=0,Rt=null,Ne={},$e=null;function F(i){Ie++,t.monitorRunDependencies?.(Ie),i?(A(!Ne[i]),Ne[i]=1,$e===null&&typeof setInterval<"u"&&($e=setInterval(()=>{if(z){clearInterval($e),$e=null;return}var s=!1;for(var c in Ne)s||(s=!0,L("still waiting on run dependencies:")),L(`dependency: ${c}`);s&&L("(end of list)")},1e4))):L("warning: run dependency added without ID")}function w(i){if(Ie--,t.monitorRunDependencies?.(Ie),i?(A(Ne[i]),delete Ne[i]):L("warning: run dependency removed without ID"),Ie==0&&($e!==null&&(clearInterval($e),$e=null),Rt)){var s=Rt;Rt=null,s()}}function $(i){t.onAbort?.(i),i="Aborted("+i+")",L(i),z=!0;var s=new WebAssembly.RuntimeError(i);throw ot?.(s),s}function re(i,s){return(...c)=>{A(Mt,`native function \`${i}\` called before runtime initialization`);var u=Li[i];return A(u,`exported native function \`${i}\` not found`),A(c.length<=s,`native function \`${i}\` called with ${c.length} args but expects ${s}`),u(...c)}}var he;function ne(){return t.locateFile?v("mujoco.wasm"):new URL(""+new URL("mujoco-Bp43jdDU.wasm",import.meta.url).href,import.meta.url).href}function ke(i){if(i==he&&N)return new Uint8Array(N);if(R)return R(i);throw"both async and sync fetching of the wasm failed"}async function Ae(i){if(!N)try{var s=await b(i);return new Uint8Array(s)}catch{}return ke(i)}async function He(i,s){try{var c=await Ae(i),u=await WebAssembly.instantiate(c,s);return u}catch(d){L(`failed to asynchronously prepare wasm: ${d}`),V(he)&&L(`warning: Loading from a file URI (${he}) is not supported in most browsers. See https://emscripten.org/docs/getting_started/FAQ.html#how-do-i-run-a-local-webserver-for-testing-why-does-my-program-stall-in-downloading-or-preparing`),$(d)}}async function Oe(i,s,c){if(!i&&typeof WebAssembly.instantiateStreaming=="function"&&!V(s)&&!o)try{var u=fetch(s,{credentials:"same-origin"}),d=await WebAssembly.instantiateStreaming(u,c);return d}catch(x){L(`wasm streaming compile failed: ${x}`),L("falling back to ArrayBuffer instantiation")}return He(s,c)}function me(){return{env:ul,wasi_snapshot_preview1:ul}}async function be(){function i(M,T){return Li=M.exports,_t=Li.memory,A(_t,"memory not found in wasm exports"),O(),xs=Li.__indirect_function_table,A(xs,"table not found in wasm exports"),ch(Li),w("wasm-instantiate"),Li}F("wasm-instantiate");var s=t;function c(M){return A(t===s,"the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?"),s=null,i(M.instance)}var u=me();if(t.instantiateWasm)return new Promise((M,T)=>{try{t.instantiateWasm(u,(I,X)=>{M(i(I,X))})}catch(I){L(`Module.instantiateWasm callback failed with error: ${I}`),T(I)}});he??=ne();var d=await Oe(N,he,u),x=c(d);return x}class je{name="ExitStatus";constructor(s){this.message=`Program terminated with exit(${s})`,this.status=s}}var We=i=>{for(;i.length>0;)i.shift()(t)},De=[],qe=i=>De.push(i),k=[],Re=i=>k.push(i),Te=!0,Se=i=>(A(typeof i=="number"),i>>>=0,"0x"+i.toString(16).padStart(8,"0")),H=i=>rl(i),B=()=>al(),Le=i=>{Le.shown||={},Le.shown[i]||(Le.shown[i]=1,o&&(i="warning: "+i),L(i))},Ke=typeof TextDecoder<"u"?new TextDecoder:void 0,vt=(i,s=0,c=NaN)=>{for(var u=s+c,d=s;i[d]&&!(d>=u);)++d;if(d-s>16&&i.buffer&&Ke)return Ke.decode(i.subarray(s,d));for(var x="";s<d;){var M=i[s++];if(!(M&128)){x+=String.fromCharCode(M);continue}var T=i[s++]&63;if((M&224)==192){x+=String.fromCharCode((M&31)<<6|T);continue}var I=i[s++]&63;if((M&240)==224?M=(M&15)<<12|T<<6|I:((M&248)!=240&&Le("Invalid UTF-8 leading byte "+Se(M)+" encountered when deserializing a UTF-8 string in wasm memory to a JS string!"),M=(M&7)<<18|T<<12|I<<6|i[s++]&63),M<65536)x+=String.fromCharCode(M);else{var X=M-65536;x+=String.fromCharCode(55296|X>>10,56320|X&1023)}}return x},ut=(i,s)=>(A(typeof i=="number",`UTF8ToString expects a number (got ${typeof i})`),i?vt(at,i,s):""),yn=(i,s,c,u)=>$(`Assertion failed: ${ut(i)}, at: `+[s?ut(s):"unknown filename",c,u?ut(u):"unknown function"]),Vt=[],Yi=0,ss=i=>{var s=new wi(i);return s.get_caught()||(s.set_caught(!0),Yi--),s.set_rethrown(!1),Vt.push(s),Ms(i),ll(i)},as=()=>{if(!Vt.length)return 0;var i=Vt[Vt.length-1];return Ms(i.excPtr),i.excPtr},Jt=0,Lr=()=>{_e(0,0),A(Vt.length>0);var i=Vt.pop();La(i.excPtr),Jt=0};class wi{constructor(s){this.excPtr=s,this.ptr=s-24}set_type(s){ye[this.ptr+4>>2]=s}get_type(){return ye[this.ptr+4>>2]}set_destructor(s){ye[this.ptr+8>>2]=s}get_destructor(){return ye[this.ptr+8>>2]}set_caught(s){s=s?1:0,Je[this.ptr+12]=s}get_caught(){return Je[this.ptr+12]!=0}set_rethrown(s){s=s?1:0,Je[this.ptr+13]=s}get_rethrown(){return Je[this.ptr+13]!=0}init(s,c){this.set_adjusted_ptr(0),this.set_type(s),this.set_destructor(c)}set_adjusted_ptr(s){ye[this.ptr+16>>2]=s}get_adjusted_ptr(){return ye[this.ptr+16>>2]}}var Ai=i=>nl(i),fi=i=>{var s=Jt?.excPtr;if(!s)return Ai(0),0;var c=new wi(s);c.set_adjusted_ptr(s);var u=c.get_type();if(!u)return Ai(0),s;for(var d of i){if(d===0||d===u)break;var x=c.ptr+16;if(cl(d,u,x))return Ai(d),s}return Ai(u),s},os=()=>fi([]),qi=i=>fi([i]),cs=(i,s)=>fi([i,s]),Fr=()=>{var i=Vt.pop();i||$("no exception to throw");var s=i.excPtr;throw i.get_rethrown()||(Vt.push(i),i.set_rethrown(!0),i.set_caught(!1),Yi++),Jt=new Y(s),Jt},ma=i=>{if(i){var s=new wi(i);Vt.push(s),s.set_rethrown(!0),Fr()}},_a=(i,s,c)=>{var u=new wi(i);throw u.init(s,c),Jt=new Y(i),Yi++,Jt},xa=()=>Yi,ga=i=>{throw Jt||(Jt=new Y(i)),Jt},Tt={isAbs:i=>i.charAt(0)==="/",splitPath:i=>{var s=/^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;return s.exec(i).slice(1)},normalizeArray:(i,s)=>{for(var c=0,u=i.length-1;u>=0;u--){var d=i[u];d==="."?i.splice(u,1):d===".."?(i.splice(u,1),c++):c&&(i.splice(u,1),c--)}if(s)for(;c;c--)i.unshift("..");return i},normalize:i=>{var s=Tt.isAbs(i),c=i.slice(-1)==="/";return i=Tt.normalizeArray(i.split("/").filter(u=>!!u),!s).join("/"),!i&&!s&&(i="."),i&&c&&(i+="/"),(s?"/":"")+i},dirname:i=>{var s=Tt.splitPath(i),c=s[0],u=s[1];return!c&&!u?".":(u&&(u=u.slice(0,-1)),c+u)},basename:i=>i&&i.match(/([^\/]+|\/)\/*$/)[1],join:(...i)=>Tt.normalize(i.join("/")),join2:(i,s)=>Tt.normalize(i+"/"+s)},va=()=>{if(o){var i=f("crypto");return s=>i.randomFillSync(s)}return s=>crypto.getRandomValues(s)},C=i=>{(C=va())(i)},W={resolve:(...i)=>{for(var s="",c=!1,u=i.length-1;u>=-1&&!c;u--){var d=u>=0?i[u]:S.cwd();if(typeof d!="string")throw new TypeError("Arguments to path.resolve must be strings");if(!d)return"";s=d+"/"+s,c=Tt.isAbs(d)}return s=Tt.normalizeArray(s.split("/").filter(x=>!!x),!c).join("/"),(c?"/":"")+s||"."},relative:(i,s)=>{i=W.resolve(i).slice(1),s=W.resolve(s).slice(1);function c(X){for(var q=0;q<X.length&&X[q]==="";q++);for(var ae=X.length-1;ae>=0&&X[ae]==="";ae--);return q>ae?[]:X.slice(q,ae-q+1)}for(var u=c(i.split("/")),d=c(s.split("/")),x=Math.min(u.length,d.length),M=x,T=0;T<x;T++)if(u[T]!==d[T]){M=T;break}for(var I=[],T=M;T<u.length;T++)I.push("..");return I=I.concat(d.slice(M)),I.join("/")}},te=[],Q=i=>{for(var s=0,c=0;c<i.length;++c){var u=i.charCodeAt(c);u<=127?s++:u<=2047?s+=2:u>=55296&&u<=57343?(s+=4,++c):s+=3}return s},j=(i,s,c,u)=>{if(A(typeof i=="string",`stringToUTF8Array expects a string (got ${typeof i})`),!(u>0))return 0;for(var d=c,x=c+u-1,M=0;M<i.length;++M){var T=i.codePointAt(M);if(T<=127){if(c>=x)break;s[c++]=T}else if(T<=2047){if(c+1>=x)break;s[c++]=192|T>>6,s[c++]=128|T&63}else if(T<=65535){if(c+2>=x)break;s[c++]=224|T>>12,s[c++]=128|T>>6&63,s[c++]=128|T&63}else{if(c+3>=x)break;T>1114111&&Le("Invalid Unicode code point "+Se(T)+" encountered when serializing a JS string to a UTF-8 string in wasm memory! (Valid unicode code points should be in range 0-0x10FFFF)."),s[c++]=240|T>>18,s[c++]=128|T>>12&63,s[c++]=128|T>>6&63,s[c++]=128|T&63,M++}}return s[c]=0,c-d},ve=(i,s,c)=>{var u=Q(i)+1,d=new Array(u),x=j(i,d,0,d.length);return d.length=x,d},Pe=()=>{if(!te.length){var i=null;if(o){var s=256,c=Buffer.alloc(s),u=0,d=process.stdin.fd;try{u=m.readSync(d,c,0,s)}catch(x){if(x.toString().includes("EOF"))u=0;else throw x}u>0&&(i=c.slice(0,u).toString("utf-8"))}else typeof window<"u"&&typeof window.prompt=="function"&&(i=window.prompt("Input: "),i!==null&&(i+=`
`));if(!i)return null;te=ve(i)}return te.shift()},Fe={ttys:[],init(){},shutdown(){},register(i,s){Fe.ttys[i]={input:[],output:[],ops:s},S.registerDevice(i,Fe.stream_ops)},stream_ops:{open(i){var s=Fe.ttys[i.node.rdev];if(!s)throw new S.ErrnoError(43);i.tty=s,i.seekable=!1},close(i){i.tty.ops.fsync(i.tty)},fsync(i){i.tty.ops.fsync(i.tty)},read(i,s,c,u,d){if(!i.tty||!i.tty.ops.get_char)throw new S.ErrnoError(60);for(var x=0,M=0;M<u;M++){var T;try{T=i.tty.ops.get_char(i.tty)}catch{throw new S.ErrnoError(29)}if(T===void 0&&x===0)throw new S.ErrnoError(6);if(T==null)break;x++,s[c+M]=T}return x&&(i.node.atime=Date.now()),x},write(i,s,c,u,d){if(!i.tty||!i.tty.ops.put_char)throw new S.ErrnoError(60);try{for(var x=0;x<u;x++)i.tty.ops.put_char(i.tty,s[c+x])}catch{throw new S.ErrnoError(29)}return u&&(i.node.mtime=i.node.ctime=Date.now()),x}},default_tty_ops:{get_char(i){return Pe()},put_char(i,s){s===null||s===10?(U(vt(i.output)),i.output=[]):s!=0&&i.output.push(s)},fsync(i){i.output?.length>0&&(U(vt(i.output)),i.output=[])},ioctl_tcgets(i){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(i,s,c){return 0},ioctl_tiocgwinsz(i){return[24,80]}},default_tty1_ops:{put_char(i,s){s===null||s===10?(L(vt(i.output)),i.output=[]):s!=0&&i.output.push(s)},fsync(i){i.output?.length>0&&(L(vt(i.output)),i.output=[])}}},Ue=i=>{$("internal error: mmapAlloc called but `emscripten_builtin_memalign` native symbol not exported")},ge={ops_table:null,mount(i){return ge.createNode(null,"/",16895,0)},createNode(i,s,c,u){if(S.isBlkdev(c)||S.isFIFO(c))throw new S.ErrnoError(63);ge.ops_table||={dir:{node:{getattr:ge.node_ops.getattr,setattr:ge.node_ops.setattr,lookup:ge.node_ops.lookup,mknod:ge.node_ops.mknod,rename:ge.node_ops.rename,unlink:ge.node_ops.unlink,rmdir:ge.node_ops.rmdir,readdir:ge.node_ops.readdir,symlink:ge.node_ops.symlink},stream:{llseek:ge.stream_ops.llseek}},file:{node:{getattr:ge.node_ops.getattr,setattr:ge.node_ops.setattr},stream:{llseek:ge.stream_ops.llseek,read:ge.stream_ops.read,write:ge.stream_ops.write,mmap:ge.stream_ops.mmap,msync:ge.stream_ops.msync}},link:{node:{getattr:ge.node_ops.getattr,setattr:ge.node_ops.setattr,readlink:ge.node_ops.readlink},stream:{}},chrdev:{node:{getattr:ge.node_ops.getattr,setattr:ge.node_ops.setattr},stream:S.chrdev_stream_ops}};var d=S.createNode(i,s,c,u);return S.isDir(d.mode)?(d.node_ops=ge.ops_table.dir.node,d.stream_ops=ge.ops_table.dir.stream,d.contents={}):S.isFile(d.mode)?(d.node_ops=ge.ops_table.file.node,d.stream_ops=ge.ops_table.file.stream,d.usedBytes=0,d.contents=null):S.isLink(d.mode)?(d.node_ops=ge.ops_table.link.node,d.stream_ops=ge.ops_table.link.stream):S.isChrdev(d.mode)&&(d.node_ops=ge.ops_table.chrdev.node,d.stream_ops=ge.ops_table.chrdev.stream),d.atime=d.mtime=d.ctime=Date.now(),i&&(i.contents[s]=d,i.atime=i.mtime=i.ctime=d.atime),d},getFileDataAsTypedArray(i){return i.contents?i.contents.subarray?i.contents.subarray(0,i.usedBytes):new Uint8Array(i.contents):new Uint8Array(0)},expandFileStorage(i,s){var c=i.contents?i.contents.length:0;if(!(c>=s)){var u=1024*1024;s=Math.max(s,c*(c<u?2:1.125)>>>0),c!=0&&(s=Math.max(s,256));var d=i.contents;i.contents=new Uint8Array(s),i.usedBytes>0&&i.contents.set(d.subarray(0,i.usedBytes),0)}},resizeFileStorage(i,s){if(i.usedBytes!=s)if(s==0)i.contents=null,i.usedBytes=0;else{var c=i.contents;i.contents=new Uint8Array(s),c&&i.contents.set(c.subarray(0,Math.min(s,i.usedBytes))),i.usedBytes=s}},node_ops:{getattr(i){var s={};return s.dev=S.isChrdev(i.mode)?i.id:1,s.ino=i.id,s.mode=i.mode,s.nlink=1,s.uid=0,s.gid=0,s.rdev=i.rdev,S.isDir(i.mode)?s.size=4096:S.isFile(i.mode)?s.size=i.usedBytes:S.isLink(i.mode)?s.size=i.link.length:s.size=0,s.atime=new Date(i.atime),s.mtime=new Date(i.mtime),s.ctime=new Date(i.ctime),s.blksize=4096,s.blocks=Math.ceil(s.size/s.blksize),s},setattr(i,s){for(const c of["mode","atime","mtime","ctime"])s[c]!=null&&(i[c]=s[c]);s.size!==void 0&&ge.resizeFileStorage(i,s.size)},lookup(i,s){throw new S.ErrnoError(44)},mknod(i,s,c,u){return ge.createNode(i,s,c,u)},rename(i,s,c){var u;try{u=S.lookupNode(s,c)}catch{}if(u){if(S.isDir(i.mode))for(var d in u.contents)throw new S.ErrnoError(55);S.hashRemoveNode(u)}delete i.parent.contents[i.name],s.contents[c]=i,i.name=c,s.ctime=s.mtime=i.parent.ctime=i.parent.mtime=Date.now()},unlink(i,s){delete i.contents[s],i.ctime=i.mtime=Date.now()},rmdir(i,s){var c=S.lookupNode(i,s);for(var u in c.contents)throw new S.ErrnoError(55);delete i.contents[s],i.ctime=i.mtime=Date.now()},readdir(i){return[".","..",...Object.keys(i.contents)]},symlink(i,s,c){var u=ge.createNode(i,s,41471,0);return u.link=c,u},readlink(i){if(!S.isLink(i.mode))throw new S.ErrnoError(28);return i.link}},stream_ops:{read(i,s,c,u,d){var x=i.node.contents;if(d>=i.node.usedBytes)return 0;var M=Math.min(i.node.usedBytes-d,u);if(A(M>=0),M>8&&x.subarray)s.set(x.subarray(d,d+M),c);else for(var T=0;T<M;T++)s[c+T]=x[d+T];return M},write(i,s,c,u,d,x){if(A(!(s instanceof ArrayBuffer)),s.buffer===Je.buffer&&(x=!1),!u)return 0;var M=i.node;if(M.mtime=M.ctime=Date.now(),s.subarray&&(!M.contents||M.contents.subarray)){if(x)return A(d===0,"canOwn must imply no weird position inside the file"),M.contents=s.subarray(c,c+u),M.usedBytes=u,u;if(M.usedBytes===0&&d===0)return M.contents=s.slice(c,c+u),M.usedBytes=u,u;if(d+u<=M.usedBytes)return M.contents.set(s.subarray(c,c+u),d),u}if(ge.expandFileStorage(M,d+u),M.contents.subarray&&s.subarray)M.contents.set(s.subarray(c,c+u),d);else for(var T=0;T<u;T++)M.contents[d+T]=s[c+T];return M.usedBytes=Math.max(M.usedBytes,d+u),u},llseek(i,s,c){var u=s;if(c===1?u+=i.position:c===2&&S.isFile(i.node.mode)&&(u+=i.node.usedBytes),u<0)throw new S.ErrnoError(28);return u},mmap(i,s,c,u,d){if(!S.isFile(i.node.mode))throw new S.ErrnoError(43);var x,M,T=i.node.contents;if(!(d&2)&&T&&T.buffer===Je.buffer)M=!1,x=T.byteOffset;else{if(M=!0,x=Ue(),!x)throw new S.ErrnoError(48);T&&((c>0||c+s<T.length)&&(T.subarray?T=T.subarray(c,c+s):T=Array.prototype.slice.call(T,c,c+s)),Je.set(T,x))}return{ptr:x,allocated:M}},msync(i,s,c,u,d){return ge.stream_ops.write(i,s,0,u,c,!1),0}}},Ye=async i=>{var s=await b(i);return A(s,`Loading data file "${i}" failed (no arrayBuffer).`),new Uint8Array(s)},ze=(...i)=>S.createDataFile(...i),ft=i=>{for(var s=i;;){if(!Ne[i])return i;i=s+Math.random()}},bt=[],kt=(i,s,c,u)=>{typeof Browser<"u"&&Browser.init();var d=!1;return bt.forEach(x=>{d||x.canHandle(s)&&(x.handle(i,s,c,u),d=!0)}),d},Nt=(i,s,c,u,d,x,M,T,I,X)=>{var q=s?W.resolve(Tt.join2(i,s)):i,ae=ft(`cp ${q}`);function ue(oe){function pe(Xe){X?.(),T||ze(i,s,Xe,u,d,I),x?.(),w(ae)}kt(oe,q,pe,()=>{M?.(),w(ae)})||pe(oe)}F(ae),typeof c=="string"?Ye(c).then(ue,M):ue(c)},Ct=i=>{var s={r:0,"r+":2,w:577,"w+":578,a:1089,"a+":1090},c=s[i];if(typeof c>"u")throw new Error(`Unknown file open mode: ${i}`);return c},Ve=(i,s)=>{var c=0;return i&&(c|=365),s&&(c|=146),c},Ot=i=>ut(tl(i)),pt={EPERM:63,ENOENT:44,ESRCH:71,EINTR:27,EIO:29,ENXIO:60,E2BIG:1,ENOEXEC:45,EBADF:8,ECHILD:12,EAGAIN:6,EWOULDBLOCK:6,ENOMEM:48,EACCES:2,EFAULT:21,ENOTBLK:105,EBUSY:10,EEXIST:20,EXDEV:75,ENODEV:43,ENOTDIR:54,EISDIR:31,EINVAL:28,ENFILE:41,EMFILE:33,ENOTTY:59,ETXTBSY:74,EFBIG:22,ENOSPC:51,ESPIPE:70,EROFS:69,EMLINK:34,EPIPE:64,EDOM:18,ERANGE:68,ENOMSG:49,EIDRM:24,ECHRNG:106,EL2NSYNC:156,EL3HLT:107,EL3RST:108,ELNRNG:109,EUNATCH:110,ENOCSI:111,EL2HLT:112,EDEADLK:16,ENOLCK:46,EBADE:113,EBADR:114,EXFULL:115,ENOANO:104,EBADRQC:103,EBADSLT:102,EDEADLOCK:16,EBFONT:101,ENOSTR:100,ENODATA:116,ETIME:117,ENOSR:118,ENONET:119,ENOPKG:120,EREMOTE:121,ENOLINK:47,EADV:122,ESRMNT:123,ECOMM:124,EPROTO:65,EMULTIHOP:36,EDOTDOT:125,EBADMSG:9,ENOTUNIQ:126,EBADFD:127,EREMCHG:128,ELIBACC:129,ELIBBAD:130,ELIBSCN:131,ELIBMAX:132,ELIBEXEC:133,ENOSYS:52,ENOTEMPTY:55,ENAMETOOLONG:37,ELOOP:32,EOPNOTSUPP:138,EPFNOSUPPORT:139,ECONNRESET:15,ENOBUFS:42,EAFNOSUPPORT:5,EPROTOTYPE:67,ENOTSOCK:57,ENOPROTOOPT:50,ESHUTDOWN:140,ECONNREFUSED:14,EADDRINUSE:3,ECONNABORTED:13,ENETUNREACH:40,ENETDOWN:38,ETIMEDOUT:73,EHOSTDOWN:142,EHOSTUNREACH:23,EINPROGRESS:26,EALREADY:7,EDESTADDRREQ:17,EMSGSIZE:35,EPROTONOSUPPORT:66,ESOCKTNOSUPPORT:137,EADDRNOTAVAIL:4,ENETRESET:39,EISCONN:30,ENOTCONN:53,ETOOMANYREFS:141,EUSERS:136,EDQUOT:19,ESTALE:72,ENOTSUP:138,ENOMEDIUM:148,EILSEQ:25,EOVERFLOW:61,ECANCELED:11,ENOTRECOVERABLE:56,EOWNERDEAD:62,ESTRPIPE:135},S={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:!1,ignorePermissions:!0,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class extends Error{name="ErrnoError";constructor(i){super(Mt?Ot(i):""),this.errno=i;for(var s in pt)if(pt[s]===i){this.code=s;break}}},FSStream:class{shared={};get object(){return this.node}set object(i){this.node=i}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(i){this.shared.flags=i}get position(){return this.shared.position}set position(i){this.shared.position=i}},FSNode:class{node_ops={};stream_ops={};readMode=365;writeMode=146;mounted=null;constructor(i,s,c,u){i||(i=this),this.parent=i,this.mount=i.mount,this.id=S.nextInode++,this.name=s,this.mode=c,this.rdev=u,this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(i){i?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(i){i?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return S.isDir(this.mode)}get isDevice(){return S.isChrdev(this.mode)}},lookupPath(i,s={}){if(!i)throw new S.ErrnoError(44);s.follow_mount??=!0,Tt.isAbs(i)||(i=S.cwd()+"/"+i);e:for(var c=0;c<40;c++){for(var u=i.split("/").filter(X=>!!X),d=S.root,x="/",M=0;M<u.length;M++){var T=M===u.length-1;if(T&&s.parent)break;if(u[M]!=="."){if(u[M]===".."){if(x=Tt.dirname(x),S.isRoot(d)){i=x+"/"+u.slice(M+1).join("/");continue e}else d=d.parent;continue}x=Tt.join2(x,u[M]);try{d=S.lookupNode(d,u[M])}catch(X){if(X?.errno===44&&T&&s.noent_okay)return{path:x};throw X}if(S.isMountpoint(d)&&(!T||s.follow_mount)&&(d=d.mounted.root),S.isLink(d.mode)&&(!T||s.follow)){if(!d.node_ops.readlink)throw new S.ErrnoError(52);var I=d.node_ops.readlink(d);Tt.isAbs(I)||(I=Tt.dirname(x)+"/"+I),i=I+"/"+u.slice(M+1).join("/");continue e}}}return{path:x,node:d}}throw new S.ErrnoError(32)},getPath(i){for(var s;;){if(S.isRoot(i)){var c=i.mount.mountpoint;return s?c[c.length-1]!=="/"?`${c}/${s}`:c+s:c}s=s?`${i.name}/${s}`:i.name,i=i.parent}},hashName(i,s){for(var c=0,u=0;u<s.length;u++)c=(c<<5)-c+s.charCodeAt(u)|0;return(i+c>>>0)%S.nameTable.length},hashAddNode(i){var s=S.hashName(i.parent.id,i.name);i.name_next=S.nameTable[s],S.nameTable[s]=i},hashRemoveNode(i){var s=S.hashName(i.parent.id,i.name);if(S.nameTable[s]===i)S.nameTable[s]=i.name_next;else for(var c=S.nameTable[s];c;){if(c.name_next===i){c.name_next=i.name_next;break}c=c.name_next}},lookupNode(i,s){var c=S.mayLookup(i);if(c)throw new S.ErrnoError(c);for(var u=S.hashName(i.id,s),d=S.nameTable[u];d;d=d.name_next){var x=d.name;if(d.parent.id===i.id&&x===s)return d}return S.lookup(i,s)},createNode(i,s,c,u){A(typeof i=="object");var d=new S.FSNode(i,s,c,u);return S.hashAddNode(d),d},destroyNode(i){S.hashRemoveNode(i)},isRoot(i){return i===i.parent},isMountpoint(i){return!!i.mounted},isFile(i){return(i&61440)===32768},isDir(i){return(i&61440)===16384},isLink(i){return(i&61440)===40960},isChrdev(i){return(i&61440)===8192},isBlkdev(i){return(i&61440)===24576},isFIFO(i){return(i&61440)===4096},isSocket(i){return(i&49152)===49152},flagsToPermissionString(i){var s=["r","w","rw"][i&3];return i&512&&(s+="w"),s},nodePermissions(i,s){return S.ignorePermissions?0:s.includes("r")&&!(i.mode&292)||s.includes("w")&&!(i.mode&146)||s.includes("x")&&!(i.mode&73)?2:0},mayLookup(i){if(!S.isDir(i.mode))return 54;var s=S.nodePermissions(i,"x");return s||(i.node_ops.lookup?0:2)},mayCreate(i,s){if(!S.isDir(i.mode))return 54;try{var c=S.lookupNode(i,s);return 20}catch{}return S.nodePermissions(i,"wx")},mayDelete(i,s,c){var u;try{u=S.lookupNode(i,s)}catch(x){return x.errno}var d=S.nodePermissions(i,"wx");if(d)return d;if(c){if(!S.isDir(u.mode))return 54;if(S.isRoot(u)||S.getPath(u)===S.cwd())return 10}else if(S.isDir(u.mode))return 31;return 0},mayOpen(i,s){return i?S.isLink(i.mode)?32:S.isDir(i.mode)&&(S.flagsToPermissionString(s)!=="r"||s&576)?31:S.nodePermissions(i,S.flagsToPermissionString(s)):44},checkOpExists(i,s){if(!i)throw new S.ErrnoError(s);return i},MAX_OPEN_FDS:4096,nextfd(){for(var i=0;i<=S.MAX_OPEN_FDS;i++)if(!S.streams[i])return i;throw new S.ErrnoError(33)},getStreamChecked(i){var s=S.getStream(i);if(!s)throw new S.ErrnoError(8);return s},getStream:i=>S.streams[i],createStream(i,s=-1){return A(s>=-1),i=Object.assign(new S.FSStream,i),s==-1&&(s=S.nextfd()),i.fd=s,S.streams[s]=i,i},closeStream(i){S.streams[i]=null},dupStream(i,s=-1){var c=S.createStream(i,s);return c.stream_ops?.dup?.(c),c},doSetAttr(i,s,c){var u=i?.stream_ops.setattr,d=u?i:s;u??=s.node_ops.setattr,S.checkOpExists(u,63),u(d,c)},chrdev_stream_ops:{open(i){var s=S.getDevice(i.node.rdev);i.stream_ops=s.stream_ops,i.stream_ops.open?.(i)},llseek(){throw new S.ErrnoError(70)}},major:i=>i>>8,minor:i=>i&255,makedev:(i,s)=>i<<8|s,registerDevice(i,s){S.devices[i]={stream_ops:s}},getDevice:i=>S.devices[i],getMounts(i){for(var s=[],c=[i];c.length;){var u=c.pop();s.push(u),c.push(...u.mounts)}return s},syncfs(i,s){typeof i=="function"&&(s=i,i=!1),S.syncFSRequests++,S.syncFSRequests>1&&L(`warning: ${S.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`);var c=S.getMounts(S.root.mount),u=0;function d(M){return A(S.syncFSRequests>0),S.syncFSRequests--,s(M)}function x(M){if(M)return x.errored?void 0:(x.errored=!0,d(M));++u>=c.length&&d(null)}c.forEach(M=>{if(!M.type.syncfs)return x(null);M.type.syncfs(M,i,x)})},mount(i,s,c){if(typeof i=="string")throw i;var u=c==="/",d=!c,x;if(u&&S.root)throw new S.ErrnoError(10);if(!u&&!d){var M=S.lookupPath(c,{follow_mount:!1});if(c=M.path,x=M.node,S.isMountpoint(x))throw new S.ErrnoError(10);if(!S.isDir(x.mode))throw new S.ErrnoError(54)}var T={type:i,opts:s,mountpoint:c,mounts:[]},I=i.mount(T);return I.mount=T,T.root=I,u?S.root=I:x&&(x.mounted=T,x.mount&&x.mount.mounts.push(T)),I},unmount(i){var s=S.lookupPath(i,{follow_mount:!1});if(!S.isMountpoint(s.node))throw new S.ErrnoError(28);var c=s.node,u=c.mounted,d=S.getMounts(u);Object.keys(S.nameTable).forEach(M=>{for(var T=S.nameTable[M];T;){var I=T.name_next;d.includes(T.mount)&&S.destroyNode(T),T=I}}),c.mounted=null;var x=c.mount.mounts.indexOf(u);A(x!==-1),c.mount.mounts.splice(x,1)},lookup(i,s){return i.node_ops.lookup(i,s)},mknod(i,s,c){var u=S.lookupPath(i,{parent:!0}),d=u.node,x=Tt.basename(i);if(!x)throw new S.ErrnoError(28);if(x==="."||x==="..")throw new S.ErrnoError(20);var M=S.mayCreate(d,x);if(M)throw new S.ErrnoError(M);if(!d.node_ops.mknod)throw new S.ErrnoError(63);return d.node_ops.mknod(d,x,s,c)},statfs(i){return S.statfsNode(S.lookupPath(i,{follow:!0}).node)},statfsStream(i){return S.statfsNode(i.node)},statfsNode(i){var s={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:S.nextInode,ffree:S.nextInode-1,fsid:42,flags:2,namelen:255};return i.node_ops.statfs&&Object.assign(s,i.node_ops.statfs(i.mount.opts.root)),s},create(i,s=438){return s&=4095,s|=32768,S.mknod(i,s,0)},mkdir(i,s=511){return s&=1023,s|=16384,S.mknod(i,s,0)},mkdirTree(i,s){var c=i.split("/"),u="";for(var d of c)if(d){(u||Tt.isAbs(i))&&(u+="/"),u+=d;try{S.mkdir(u,s)}catch(x){if(x.errno!=20)throw x}}},mkdev(i,s,c){return typeof c>"u"&&(c=s,s=438),s|=8192,S.mknod(i,s,c)},symlink(i,s){if(!W.resolve(i))throw new S.ErrnoError(44);var c=S.lookupPath(s,{parent:!0}),u=c.node;if(!u)throw new S.ErrnoError(44);var d=Tt.basename(s),x=S.mayCreate(u,d);if(x)throw new S.ErrnoError(x);if(!u.node_ops.symlink)throw new S.ErrnoError(63);return u.node_ops.symlink(u,d,i)},rename(i,s){var c=Tt.dirname(i),u=Tt.dirname(s),d=Tt.basename(i),x=Tt.basename(s),M,T,I;if(M=S.lookupPath(i,{parent:!0}),T=M.node,M=S.lookupPath(s,{parent:!0}),I=M.node,!T||!I)throw new S.ErrnoError(44);if(T.mount!==I.mount)throw new S.ErrnoError(75);var X=S.lookupNode(T,d),q=W.relative(i,u);if(q.charAt(0)!==".")throw new S.ErrnoError(28);if(q=W.relative(s,c),q.charAt(0)!==".")throw new S.ErrnoError(55);var ae;try{ae=S.lookupNode(I,x)}catch{}if(X!==ae){var ue=S.isDir(X.mode),oe=S.mayDelete(T,d,ue);if(oe)throw new S.ErrnoError(oe);if(oe=ae?S.mayDelete(I,x,ue):S.mayCreate(I,x),oe)throw new S.ErrnoError(oe);if(!T.node_ops.rename)throw new S.ErrnoError(63);if(S.isMountpoint(X)||ae&&S.isMountpoint(ae))throw new S.ErrnoError(10);if(I!==T&&(oe=S.nodePermissions(T,"w"),oe))throw new S.ErrnoError(oe);S.hashRemoveNode(X);try{T.node_ops.rename(X,I,x),X.parent=I}catch(pe){throw pe}finally{S.hashAddNode(X)}}},rmdir(i){var s=S.lookupPath(i,{parent:!0}),c=s.node,u=Tt.basename(i),d=S.lookupNode(c,u),x=S.mayDelete(c,u,!0);if(x)throw new S.ErrnoError(x);if(!c.node_ops.rmdir)throw new S.ErrnoError(63);if(S.isMountpoint(d))throw new S.ErrnoError(10);c.node_ops.rmdir(c,u),S.destroyNode(d)},readdir(i){var s=S.lookupPath(i,{follow:!0}),c=s.node,u=S.checkOpExists(c.node_ops.readdir,54);return u(c)},unlink(i){var s=S.lookupPath(i,{parent:!0}),c=s.node;if(!c)throw new S.ErrnoError(44);var u=Tt.basename(i),d=S.lookupNode(c,u),x=S.mayDelete(c,u,!1);if(x)throw new S.ErrnoError(x);if(!c.node_ops.unlink)throw new S.ErrnoError(63);if(S.isMountpoint(d))throw new S.ErrnoError(10);c.node_ops.unlink(c,u),S.destroyNode(d)},readlink(i){var s=S.lookupPath(i),c=s.node;if(!c)throw new S.ErrnoError(44);if(!c.node_ops.readlink)throw new S.ErrnoError(28);return c.node_ops.readlink(c)},stat(i,s){var c=S.lookupPath(i,{follow:!s}),u=c.node,d=S.checkOpExists(u.node_ops.getattr,63);return d(u)},fstat(i){var s=S.getStreamChecked(i),c=s.node,u=s.stream_ops.getattr,d=u?s:c;return u??=c.node_ops.getattr,S.checkOpExists(u,63),u(d)},lstat(i){return S.stat(i,!0)},doChmod(i,s,c,u){S.doSetAttr(i,s,{mode:c&4095|s.mode&-4096,ctime:Date.now(),dontFollow:u})},chmod(i,s,c){var u;if(typeof i=="string"){var d=S.lookupPath(i,{follow:!c});u=d.node}else u=i;S.doChmod(null,u,s,c)},lchmod(i,s){S.chmod(i,s,!0)},fchmod(i,s){var c=S.getStreamChecked(i);S.doChmod(c,c.node,s,!1)},doChown(i,s,c){S.doSetAttr(i,s,{timestamp:Date.now(),dontFollow:c})},chown(i,s,c,u){var d;if(typeof i=="string"){var x=S.lookupPath(i,{follow:!u});d=x.node}else d=i;S.doChown(null,d,u)},lchown(i,s,c){S.chown(i,s,c,!0)},fchown(i,s,c){var u=S.getStreamChecked(i);S.doChown(u,u.node,!1)},doTruncate(i,s,c){if(S.isDir(s.mode))throw new S.ErrnoError(31);if(!S.isFile(s.mode))throw new S.ErrnoError(28);var u=S.nodePermissions(s,"w");if(u)throw new S.ErrnoError(u);S.doSetAttr(i,s,{size:c,timestamp:Date.now()})},truncate(i,s){if(s<0)throw new S.ErrnoError(28);var c;if(typeof i=="string"){var u=S.lookupPath(i,{follow:!0});c=u.node}else c=i;S.doTruncate(null,c,s)},ftruncate(i,s){var c=S.getStreamChecked(i);if(s<0||(c.flags&2097155)===0)throw new S.ErrnoError(28);S.doTruncate(c,c.node,s)},utime(i,s,c){var u=S.lookupPath(i,{follow:!0}),d=u.node,x=S.checkOpExists(d.node_ops.setattr,63);x(d,{atime:s,mtime:c})},open(i,s,c=438){if(i==="")throw new S.ErrnoError(44);s=typeof s=="string"?Ct(s):s,s&64?c=c&4095|32768:c=0;var u,d;if(typeof i=="object")u=i;else{d=i.endsWith("/");var x=S.lookupPath(i,{follow:!(s&131072),noent_okay:!0});u=x.node,i=x.path}var M=!1;if(s&64)if(u){if(s&128)throw new S.ErrnoError(20)}else{if(d)throw new S.ErrnoError(31);u=S.mknod(i,c|511,0),M=!0}if(!u)throw new S.ErrnoError(44);if(S.isChrdev(u.mode)&&(s&=-513),s&65536&&!S.isDir(u.mode))throw new S.ErrnoError(54);if(!M){var T=S.mayOpen(u,s);if(T)throw new S.ErrnoError(T)}s&512&&!M&&S.truncate(u,0),s&=-131713;var I=S.createStream({node:u,path:S.getPath(u),flags:s,seekable:!0,position:0,stream_ops:u.stream_ops,ungotten:[],error:!1});return I.stream_ops.open&&I.stream_ops.open(I),M&&S.chmod(u,c&511),t.logReadFiles&&!(s&1)&&(i in S.readFiles||(S.readFiles[i]=1)),I},close(i){if(S.isClosed(i))throw new S.ErrnoError(8);i.getdents&&(i.getdents=null);try{i.stream_ops.close&&i.stream_ops.close(i)}catch(s){throw s}finally{S.closeStream(i.fd)}i.fd=null},isClosed(i){return i.fd===null},llseek(i,s,c){if(S.isClosed(i))throw new S.ErrnoError(8);if(!i.seekable||!i.stream_ops.llseek)throw new S.ErrnoError(70);if(c!=0&&c!=1&&c!=2)throw new S.ErrnoError(28);return i.position=i.stream_ops.llseek(i,s,c),i.ungotten=[],i.position},read(i,s,c,u,d){if(A(c>=0),u<0||d<0)throw new S.ErrnoError(28);if(S.isClosed(i))throw new S.ErrnoError(8);if((i.flags&2097155)===1)throw new S.ErrnoError(8);if(S.isDir(i.node.mode))throw new S.ErrnoError(31);if(!i.stream_ops.read)throw new S.ErrnoError(28);var x=typeof d<"u";if(!x)d=i.position;else if(!i.seekable)throw new S.ErrnoError(70);var M=i.stream_ops.read(i,s,c,u,d);return x||(i.position+=M),M},write(i,s,c,u,d,x){if(A(c>=0),u<0||d<0)throw new S.ErrnoError(28);if(S.isClosed(i))throw new S.ErrnoError(8);if((i.flags&2097155)===0)throw new S.ErrnoError(8);if(S.isDir(i.node.mode))throw new S.ErrnoError(31);if(!i.stream_ops.write)throw new S.ErrnoError(28);i.seekable&&i.flags&1024&&S.llseek(i,0,2);var M=typeof d<"u";if(!M)d=i.position;else if(!i.seekable)throw new S.ErrnoError(70);var T=i.stream_ops.write(i,s,c,u,d,x);return M||(i.position+=T),T},mmap(i,s,c,u,d){if((u&2)!==0&&(d&2)===0&&(i.flags&2097155)!==2)throw new S.ErrnoError(2);if((i.flags&2097155)===1)throw new S.ErrnoError(2);if(!i.stream_ops.mmap)throw new S.ErrnoError(43);if(!s)throw new S.ErrnoError(28);return i.stream_ops.mmap(i,s,c,u,d)},msync(i,s,c,u,d){return A(c>=0),i.stream_ops.msync?i.stream_ops.msync(i,s,c,u,d):0},ioctl(i,s,c){if(!i.stream_ops.ioctl)throw new S.ErrnoError(59);return i.stream_ops.ioctl(i,s,c)},readFile(i,s={}){if(s.flags=s.flags||0,s.encoding=s.encoding||"binary",s.encoding!=="utf8"&&s.encoding!=="binary")throw new Error(`Invalid encoding type "${s.encoding}"`);var c=S.open(i,s.flags),u=S.stat(i),d=u.size,x=new Uint8Array(d);return S.read(c,x,0,d,0),s.encoding==="utf8"&&(x=vt(x)),S.close(c),x},writeFile(i,s,c={}){c.flags=c.flags||577;var u=S.open(i,c.flags,c.mode);if(typeof s=="string"&&(s=new Uint8Array(ve(s))),ArrayBuffer.isView(s))S.write(u,s,0,s.byteLength,void 0,c.canOwn);else throw new Error("Unsupported data type");S.close(u)},cwd:()=>S.currentPath,chdir(i){var s=S.lookupPath(i,{follow:!0});if(s.node===null)throw new S.ErrnoError(44);if(!S.isDir(s.node.mode))throw new S.ErrnoError(54);var c=S.nodePermissions(s.node,"x");if(c)throw new S.ErrnoError(c);S.currentPath=s.path},createDefaultDirectories(){S.mkdir("/tmp"),S.mkdir("/home"),S.mkdir("/home/web_user")},createDefaultDevices(){S.mkdir("/dev"),S.registerDevice(S.makedev(1,3),{read:()=>0,write:(u,d,x,M,T)=>M,llseek:()=>0}),S.mkdev("/dev/null",S.makedev(1,3)),Fe.register(S.makedev(5,0),Fe.default_tty_ops),Fe.register(S.makedev(6,0),Fe.default_tty1_ops),S.mkdev("/dev/tty",S.makedev(5,0)),S.mkdev("/dev/tty1",S.makedev(6,0));var i=new Uint8Array(1024),s=0,c=()=>(s===0&&(C(i),s=i.byteLength),i[--s]);S.createDevice("/dev","random",c),S.createDevice("/dev","urandom",c),S.mkdir("/dev/shm"),S.mkdir("/dev/shm/tmp")},createSpecialDirectories(){S.mkdir("/proc");var i=S.mkdir("/proc/self");S.mkdir("/proc/self/fd"),S.mount({mount(){var s=S.createNode(i,"fd",16895,73);return s.stream_ops={llseek:ge.stream_ops.llseek},s.node_ops={lookup(c,u){var d=+u,x=S.getStreamChecked(d),M={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>x.path},id:d+1};return M.parent=M,M},readdir(){return Array.from(S.streams.entries()).filter(([c,u])=>u).map(([c,u])=>c.toString())}},s}},{},"/proc/self/fd")},createStandardStreams(i,s,c){i?S.createDevice("/dev","stdin",i):S.symlink("/dev/tty","/dev/stdin"),s?S.createDevice("/dev","stdout",null,s):S.symlink("/dev/tty","/dev/stdout"),c?S.createDevice("/dev","stderr",null,c):S.symlink("/dev/tty1","/dev/stderr");var u=S.open("/dev/stdin",0),d=S.open("/dev/stdout",1),x=S.open("/dev/stderr",1);A(u.fd===0,`invalid handle for stdin (${u.fd})`),A(d.fd===1,`invalid handle for stdout (${d.fd})`),A(x.fd===2,`invalid handle for stderr (${x.fd})`)},staticInit(){S.nameTable=new Array(4096),S.mount(ge,{},"/"),S.createDefaultDirectories(),S.createDefaultDevices(),S.createSpecialDirectories(),S.filesystems={MEMFS:ge}},init(i,s,c){A(!S.initialized,"FS.init was previously called. If you want to initialize later with custom parameters, remove any earlier calls (note that one is automatically added to the generated code)"),S.initialized=!0,i??=t.stdin,s??=t.stdout,c??=t.stderr,S.createStandardStreams(i,s,c)},quit(){S.initialized=!1,Pa(0);for(var i of S.streams)i&&S.close(i)},findObject(i,s){var c=S.analyzePath(i,s);return c.exists?c.object:null},analyzePath(i,s){try{var c=S.lookupPath(i,{follow:!s});i=c.path}catch{}var u={isRoot:!1,exists:!1,error:0,name:null,path:null,object:null,parentExists:!1,parentPath:null,parentObject:null};try{var c=S.lookupPath(i,{parent:!0});u.parentExists=!0,u.parentPath=c.path,u.parentObject=c.node,u.name=Tt.basename(i),c=S.lookupPath(i,{follow:!s}),u.exists=!0,u.path=c.path,u.object=c.node,u.name=c.node.name,u.isRoot=c.path==="/"}catch(d){u.error=d.errno}return u},createPath(i,s,c,u){i=typeof i=="string"?i:S.getPath(i);for(var d=s.split("/").reverse();d.length;){var x=d.pop();if(x){var M=Tt.join2(i,x);try{S.mkdir(M)}catch(T){if(T.errno!=20)throw T}i=M}}return M},createFile(i,s,c,u,d){var x=Tt.join2(typeof i=="string"?i:S.getPath(i),s),M=Ve(u,d);return S.create(x,M)},createDataFile(i,s,c,u,d,x){var M=s;i&&(i=typeof i=="string"?i:S.getPath(i),M=s?Tt.join2(i,s):i);var T=Ve(u,d),I=S.create(M,T);if(c){if(typeof c=="string"){for(var X=new Array(c.length),q=0,ae=c.length;q<ae;++q)X[q]=c.charCodeAt(q);c=X}S.chmod(I,T|146);var ue=S.open(I,577);S.write(ue,c,0,c.length,0,x),S.close(ue),S.chmod(I,T)}},createDevice(i,s,c,u){var d=Tt.join2(typeof i=="string"?i:S.getPath(i),s),x=Ve(!!c,!!u);S.createDevice.major??=64;var M=S.makedev(S.createDevice.major++,0);return S.registerDevice(M,{open(T){T.seekable=!1},close(T){u?.buffer?.length&&u(10)},read(T,I,X,q,ae){for(var ue=0,oe=0;oe<q;oe++){var pe;try{pe=c()}catch{throw new S.ErrnoError(29)}if(pe===void 0&&ue===0)throw new S.ErrnoError(6);if(pe==null)break;ue++,I[X+oe]=pe}return ue&&(T.node.atime=Date.now()),ue},write(T,I,X,q,ae){for(var ue=0;ue<q;ue++)try{u(I[X+ue])}catch{throw new S.ErrnoError(29)}return q&&(T.node.mtime=T.node.ctime=Date.now()),ue}}),S.mkdev(d,x,M)},forceLoadFile(i){if(i.isDevice||i.isFolder||i.link||i.contents)return!0;if(typeof XMLHttpRequest<"u")throw new Error("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.");try{i.contents=R(i.url),i.usedBytes=i.contents.length}catch{throw new S.ErrnoError(29)}},createLazyFile(i,s,c,u,d){class x{lengthKnown=!1;chunks=[];get(oe){if(!(oe>this.length-1||oe<0)){var pe=oe%this.chunkSize,Xe=oe/this.chunkSize|0;return this.getter(Xe)[pe]}}setDataGetter(oe){this.getter=oe}cacheLength(){var oe=new XMLHttpRequest;if(oe.open("HEAD",c,!1),oe.send(null),!(oe.status>=200&&oe.status<300||oe.status===304))throw new Error("Couldn't load "+c+". Status: "+oe.status);var pe=Number(oe.getResponseHeader("Content-length")),Xe,ht=(Xe=oe.getResponseHeader("Accept-Ranges"))&&Xe==="bytes",rt=(Xe=oe.getResponseHeader("Content-Encoding"))&&Xe==="gzip",Ut=1024*1024;ht||(Ut=pe);var gt=($t,cn)=>{if($t>cn)throw new Error("invalid range ("+$t+", "+cn+") or no bytes requested!");if(cn>pe-1)throw new Error("only "+pe+" bytes available! programmer error!");var Lt=new XMLHttpRequest;if(Lt.open("GET",c,!1),pe!==Ut&&Lt.setRequestHeader("Range","bytes="+$t+"-"+cn),Lt.responseType="arraybuffer",Lt.overrideMimeType&&Lt.overrideMimeType("text/plain; charset=x-user-defined"),Lt.send(null),!(Lt.status>=200&&Lt.status<300||Lt.status===304))throw new Error("Couldn't load "+c+". Status: "+Lt.status);return Lt.response!==void 0?new Uint8Array(Lt.response||[]):ve(Lt.responseText||"")},rn=this;rn.setDataGetter($t=>{var cn=$t*Ut,Lt=($t+1)*Ut-1;if(Lt=Math.min(Lt,pe-1),typeof rn.chunks[$t]>"u"&&(rn.chunks[$t]=gt(cn,Lt)),typeof rn.chunks[$t]>"u")throw new Error("doXHR failed!");return rn.chunks[$t]}),(rt||!pe)&&(Ut=pe=1,pe=this.getter(0).length,Ut=pe,U("LazyFiles on gzip forces download of the whole file when length is accessed")),this._length=pe,this._chunkSize=Ut,this.lengthKnown=!0}get length(){return this.lengthKnown||this.cacheLength(),this._length}get chunkSize(){return this.lengthKnown||this.cacheLength(),this._chunkSize}}if(typeof XMLHttpRequest<"u"){if(!a)throw"Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc";var M=new x,T={isDevice:!1,contents:M}}else var T={isDevice:!1,url:c};var I=S.createFile(i,s,T,u,d);T.contents?I.contents=T.contents:T.url&&(I.contents=null,I.url=T.url),Object.defineProperties(I,{usedBytes:{get:function(){return this.contents.length}}});var X={},q=Object.keys(I.stream_ops);q.forEach(ue=>{var oe=I.stream_ops[ue];X[ue]=(...pe)=>(S.forceLoadFile(I),oe(...pe))});function ae(ue,oe,pe,Xe,ht){var rt=ue.node.contents;if(ht>=rt.length)return 0;var Ut=Math.min(rt.length-ht,Xe);if(A(Ut>=0),rt.slice)for(var gt=0;gt<Ut;gt++)oe[pe+gt]=rt[ht+gt];else for(var gt=0;gt<Ut;gt++)oe[pe+gt]=rt.get(ht+gt);return Ut}return X.read=(ue,oe,pe,Xe,ht)=>(S.forceLoadFile(I),ae(ue,oe,pe,Xe,ht)),X.mmap=(ue,oe,pe,Xe,ht)=>{S.forceLoadFile(I);var rt=Ue();if(!rt)throw new S.ErrnoError(48);return ae(ue,Je,rt,oe,pe),{ptr:rt,allocated:!0}},I.stream_ops=X,I},absolutePath(){$("FS.absolutePath has been removed; use PATH_FS.resolve instead")},createFolder(){$("FS.createFolder has been removed; use FS.mkdir instead")},createLink(){$("FS.createLink has been removed; use FS.symlink instead")},joinPath(){$("FS.joinPath has been removed; use PATH.join instead")},mmapAlloc(){$("FS.mmapAlloc has been replaced by the top level function mmapAlloc")},standardizePath(){$("FS.standardizePath has been removed; use PATH.normalize instead")}},Et={DEFAULT_POLLMASK:5,calculateAt(i,s,c){if(Tt.isAbs(s))return s;var u;if(i===-100)u=S.cwd();else{var d=Et.getStreamFromFD(i);u=d.path}if(s.length==0){if(!c)throw new S.ErrnoError(44);return u}return u+"/"+s},writeStat(i,s){ie[i>>2]=s.dev,ie[i+4>>2]=s.mode,ye[i+8>>2]=s.nlink,ie[i+12>>2]=s.uid,ie[i+16>>2]=s.gid,ie[i+20>>2]=s.rdev,Pt[i+24>>3]=BigInt(s.size),ie[i+32>>2]=4096,ie[i+36>>2]=s.blocks;var c=s.atime.getTime(),u=s.mtime.getTime(),d=s.ctime.getTime();return Pt[i+40>>3]=BigInt(Math.floor(c/1e3)),ye[i+48>>2]=c%1e3*1e3*1e3,Pt[i+56>>3]=BigInt(Math.floor(u/1e3)),ye[i+64>>2]=u%1e3*1e3*1e3,Pt[i+72>>3]=BigInt(Math.floor(d/1e3)),ye[i+80>>2]=d%1e3*1e3*1e3,Pt[i+88>>3]=BigInt(s.ino),0},writeStatFs(i,s){ie[i+4>>2]=s.bsize,ie[i+40>>2]=s.bsize,ie[i+8>>2]=s.blocks,ie[i+12>>2]=s.bfree,ie[i+16>>2]=s.bavail,ie[i+20>>2]=s.files,ie[i+24>>2]=s.ffree,ie[i+28>>2]=s.fsid,ie[i+44>>2]=s.flags,ie[i+36>>2]=s.namelen},doMsync(i,s,c,u,d){if(!S.isFile(s.node.mode))throw new S.ErrnoError(43);if(u&2)return 0;var x=at.slice(i,i+c);S.msync(s,x,d,c,u)},getStreamFromFD(i){var s=S.getStreamChecked(i);return s},varargs:void 0,getStr(i){var s=ut(i);return s}};function on(i,s,c){try{var u=Et.getStreamFromFD(i);if(A(!c),u.fd===s)return-28;if(s<0||s>=S.MAX_OPEN_FDS)return-8;var d=S.getStream(s);return d&&S.close(d),S.dupStream(u,s).fd}catch(x){if(typeof S>"u"||x.name!=="ErrnoError")throw x;return-x.errno}}var On=()=>{A(Et.varargs!=null);var i=ie[+Et.varargs>>2];return Et.varargs+=4,i},wt=On;function tn(i,s,c){Et.varargs=c;try{var u=Et.getStreamFromFD(i);switch(s){case 0:{var d=On();if(d<0)return-28;for(;S.streams[d];)d++;var x;return x=S.dupStream(u,d),x.fd}case 1:case 2:return 0;case 3:return u.flags;case 4:{var d=On();return u.flags|=d,0}case 12:{var d=wt(),M=0;return ce[d+M>>1]=2,0}case 13:case 14:return 0}return-28}catch(T){if(typeof S>"u"||T.name!=="ErrnoError")throw T;return-T.errno}}function dn(i,s){try{return Et.writeStat(s,S.fstat(i))}catch(c){if(typeof S>"u"||c.name!=="ErrnoError")throw c;return-c.errno}}function Qt(i,s,c){Et.varargs=c;try{var u=Et.getStreamFromFD(i);switch(s){case 21509:return u.tty?0:-59;case 21505:{if(!u.tty)return-59;if(u.tty.ops.ioctl_tcgets){var d=u.tty.ops.ioctl_tcgets(u),x=wt();ie[x>>2]=d.c_iflag||0,ie[x+4>>2]=d.c_oflag||0,ie[x+8>>2]=d.c_cflag||0,ie[x+12>>2]=d.c_lflag||0;for(var M=0;M<32;M++)Je[x+M+17]=d.c_cc[M]||0;return 0}return 0}case 21510:case 21511:case 21512:return u.tty?0:-59;case 21506:case 21507:case 21508:{if(!u.tty)return-59;if(u.tty.ops.ioctl_tcsets){for(var x=wt(),T=ie[x>>2],I=ie[x+4>>2],X=ie[x+8>>2],q=ie[x+12>>2],ae=[],M=0;M<32;M++)ae.push(Je[x+M+17]);return u.tty.ops.ioctl_tcsets(u.tty,s,{c_iflag:T,c_oflag:I,c_cflag:X,c_lflag:q,c_cc:ae})}return 0}case 21519:{if(!u.tty)return-59;var x=wt();return ie[x>>2]=0,0}case 21520:return u.tty?-28:-59;case 21531:{var x=wt();return S.ioctl(u,s,x)}case 21523:{if(!u.tty)return-59;if(u.tty.ops.ioctl_tiocgwinsz){var ue=u.tty.ops.ioctl_tiocgwinsz(u.tty),x=wt();ce[x>>1]=ue[0],ce[x+2>>1]=ue[1]}return 0}case 21524:return u.tty?0:-59;case 21515:return u.tty?0:-59;default:return-28}}catch(oe){if(typeof S>"u"||oe.name!=="ErrnoError")throw oe;return-oe.errno}}function nn(i,s){try{return i=Et.getStr(i),Et.writeStat(s,S.lstat(i))}catch(c){if(typeof S>"u"||c.name!=="ErrnoError")throw c;return-c.errno}}function Ir(i,s,c,u){try{s=Et.getStr(s);var d=u&256,x=u&4096;return u=u&-6401,A(!u,`unknown flags in __syscall_newfstatat: ${u}`),s=Et.calculateAt(i,s,x),Et.writeStat(c,d?S.lstat(s):S.stat(s))}catch(M){if(typeof S>"u"||M.name!=="ErrnoError")throw M;return-M.errno}}function Yn(i,s,c,u){Et.varargs=u;try{s=Et.getStr(s),s=Et.calculateAt(i,s);var d=u?On():0;return S.open(s,c,d).fd}catch(x){if(typeof S>"u"||x.name!=="ErrnoError")throw x;return-x.errno}}function rf(i,s){try{return i=Et.getStr(i),Et.writeStat(s,S.stat(i))}catch(c){if(typeof S>"u"||c.name!=="ErrnoError")throw c;return-c.errno}}var sf=()=>$("native code called abort()"),Wt=i=>{for(var s="";;){var c=at[i++];if(!c)return s;s+=String.fromCharCode(c)}},Ki={},Ri={},ls={},Ur=class extends Error{constructor(s){super(s),this.name="BindingError"}},mt=i=>{throw new Ur(i)};function af(i,s,c={}){var u=s.name;if(i||mt(`type "${u}" must have a positive integer typeid pointer`),Ri.hasOwnProperty(i)){if(c.ignoreDuplicateRegistrations)return;mt(`Cannot register type '${u}' twice`)}if(Ri[i]=s,delete ls[i],Ki.hasOwnProperty(i)){var d=Ki[i];delete Ki[i],d.forEach(x=>x())}}function Sn(i,s,c={}){if(s.argPackAdvance===void 0)throw new TypeError("registerType registeredInstance requires argPackAdvance");return af(i,s,c)}var wc=(i,s,c)=>{switch(s){case 1:return c?u=>Je[u]:u=>at[u];case 2:return c?u=>ce[u>>1]:u=>de[u>>1];case 4:return c?u=>ie[u>>2]:u=>ye[u>>2];case 8:return c?u=>Pt[u>>3]:u=>it[u>>3];default:throw new TypeError(`invalid integer width (${s}): ${i}`)}},Ci=i=>{if(i===null)return"null";var s=typeof i;return s==="object"||s==="array"||s==="function"?i.toString():""+i},Ac=(i,s,c,u)=>{if(s<c||s>u)throw new TypeError(`Passing a number "${Ci(s)}" from JS side to C/C++ side to an argument of type "${i}", which is outside the valid range [${c}, ${u}]!`)},of=(i,s,c,u,d)=>{s=Wt(s);const x=u===0n;let M=T=>T;if(x){const T=c*8;M=I=>BigInt.asUintN(T,I),d=M(d)}Sn(i,{name:s,fromWireType:M,toWireType:(T,I)=>{if(typeof I=="number")I=BigInt(I);else if(typeof I!="bigint")throw new TypeError(`Cannot convert "${Ci(I)}" to ${this.name}`);return Ac(s,I,u,d),I},argPackAdvance:kn,readValueFromPointer:wc(s,c,!x),destructorFunction:null})},kn=8,cf=(i,s,c,u)=>{s=Wt(s),Sn(i,{name:s,fromWireType:function(d){return!!d},toWireType:function(d,x){return x?c:u},argPackAdvance:kn,readValueFromPointer:function(d){return this.fromWireType(at[d])},destructorFunction:null})},lf=i=>({count:i.count,deleteScheduled:i.deleteScheduled,preservePointerOnDelete:i.preservePointerOnDelete,ptr:i.ptr,ptrType:i.ptrType,smartPtr:i.smartPtr,smartPtrType:i.smartPtrType}),ya=i=>{function s(c){return c.$$.ptrType.registeredClass.name}mt(s(i)+" instance already deleted")},Sa=!1,Rc=i=>{},uf=i=>{i.smartPtr?i.smartPtrType.rawDestructor(i.smartPtr):i.ptrType.registeredClass.rawDestructor(i.ptr)},Cc=i=>{i.count.value-=1;var s=i.count.value===0;s&&uf(i)},Pc=(i,s,c)=>{if(s===c)return i;if(c.baseClass===void 0)return null;var u=Pc(i,s,c.baseClass);return u===null?null:c.downcast(u)},Dc={},ff={},df=(i,s)=>{for(s===void 0&&mt("ptr should not be undefined");i.baseClass;)s=i.upcast(s),i=i.baseClass;return s},hf=(i,s)=>(s=df(i,s),ff[s]),pf=class extends Error{constructor(s){super(s),this.name="InternalError"}},us=i=>{throw new pf(i)},fs=(i,s)=>{(!s.ptrType||!s.ptr)&&us("makeClassHandle requires ptr and ptrType");var c=!!s.smartPtrType,u=!!s.smartPtr;return c!==u&&us("Both smartPtrType and smartPtr must be specified"),s.count={value:1},Nr(Object.create(i,{$$:{value:s,writable:!0}}))};function Lc(i){var s=this.getPointee(i);if(!s)return this.destructor(i),null;var c=hf(this.registeredClass,s);if(c!==void 0){if(c.$$.count.value===0)return c.$$.ptr=s,c.$$.smartPtr=i,c.clone();var u=c.clone();return this.destructor(i),u}function d(){return this.isSmartPointer?fs(this.registeredClass.instancePrototype,{ptrType:this.pointeeType,ptr:s,smartPtrType:this,smartPtr:i}):fs(this.registeredClass.instancePrototype,{ptrType:this,ptr:i})}var x=this.registeredClass.getActualType(s),M=Dc[x];if(!M)return d.call(this);var T;this.isConst?T=M.constPointerType:T=M.pointerType;var I=Pc(s,this.registeredClass,T.registeredClass);return I===null?d.call(this):this.isSmartPointer?fs(T.registeredClass.instancePrototype,{ptrType:T,ptr:I,smartPtrType:this,smartPtr:i}):fs(T.registeredClass.instancePrototype,{ptrType:T,ptr:I})}var Nr=i=>typeof FinalizationRegistry>"u"?(Nr=s=>s,i):(Sa=new FinalizationRegistry(s=>{console.warn(s.leakWarning),Cc(s.$$)}),Nr=s=>{var c=s.$$,u=!!c.smartPtr;if(u){var d={$$:c},x=c.ptrType.registeredClass,M=new Error(`Embind found a leaked C++ instance ${x.name} <${Se(c.ptr)}>.
We'll free it automatically in this case, but this functionality is not reliable across various environments.
Make sure to invoke .delete() manually once you're done with the instance instead.
Originally allocated`);"captureStackTrace"in Error&&Error.captureStackTrace(M,Lc),d.leakWarning=M.stack.replace(/^Error: /,""),Sa.register(s,d,s)}return s},Rc=s=>Sa.unregister(s),Nr(i)),mf=()=>{let i=ds.prototype;Object.assign(i,{isAliasOf(c){if(!(this instanceof ds)||!(c instanceof ds))return!1;var u=this.$$.ptrType.registeredClass,d=this.$$.ptr;c.$$=c.$$;for(var x=c.$$.ptrType.registeredClass,M=c.$$.ptr;u.baseClass;)d=u.upcast(d),u=u.baseClass;for(;x.baseClass;)M=x.upcast(M),x=x.baseClass;return u===x&&d===M},clone(){if(this.$$.ptr||ya(this),this.$$.preservePointerOnDelete)return this.$$.count.value+=1,this;var c=Nr(Object.create(Object.getPrototypeOf(this),{$$:{value:lf(this.$$)}}));return c.$$.count.value+=1,c.$$.deleteScheduled=!1,c},delete(){this.$$.ptr||ya(this),this.$$.deleteScheduled&&!this.$$.preservePointerOnDelete&&mt("Object already scheduled for deletion"),Rc(this),Cc(this.$$),this.$$.preservePointerOnDelete||(this.$$.smartPtr=void 0,this.$$.ptr=void 0)},isDeleted(){return!this.$$.ptr},deleteLater(){return this.$$.ptr||ya(this),this.$$.deleteScheduled&&!this.$$.preservePointerOnDelete&&mt("Object already scheduled for deletion"),this.$$.deleteScheduled=!0,this}});const s=Symbol.dispose;s&&(i[s]=i.delete)};function ds(){}var hs=(i,s)=>Object.defineProperty(s,"name",{value:i}),ba=(i,s,c)=>{if(i[s].overloadTable===void 0){var u=i[s];i[s]=function(...d){return i[s].overloadTable.hasOwnProperty(d.length)||mt(`Function '${c}' called with an invalid number of arguments (${d.length}) - expects one of (${i[s].overloadTable})!`),i[s].overloadTable[d.length].apply(this,d)},i[s].overloadTable=[],i[s].overloadTable[u.argCount]=u}},Ea=(i,s,c)=>{t.hasOwnProperty(i)?((c===void 0||t[i].overloadTable!==void 0&&t[i].overloadTable[c]!==void 0)&&mt(`Cannot register public name '${i}' twice`),ba(t,i,i),t[i].overloadTable.hasOwnProperty(c)&&mt(`Cannot register multiple overloads of a function with the same number of arguments (${c})!`),t[i].overloadTable[c]=s):(t[i]=s,t[i].argCount=c)},_f=48,xf=57,gf=i=>{A(typeof i=="string"),i=i.replace(/[^a-zA-Z0-9_]/g,"$");var s=i.charCodeAt(0);return s>=_f&&s<=xf?`_${i}`:i};function vf(i,s,c,u,d,x,M,T){this.name=i,this.constructor=s,this.instancePrototype=c,this.rawDestructor=u,this.baseClass=d,this.getActualType=x,this.upcast=M,this.downcast=T,this.pureVirtualFunctions=[]}var ps=(i,s,c)=>{for(;s!==c;)s.upcast||mt(`Expected null or instance of ${c.name}, got an instance of ${s.name}`),i=s.upcast(i),s=s.baseClass;return i};function yf(i,s){if(s===null)return this.isReference&&mt(`null is not a valid ${this.name}`),0;s.$$||mt(`Cannot pass "${Ci(s)}" as a ${this.name}`),s.$$.ptr||mt(`Cannot pass deleted object as a pointer of type ${this.name}`);var c=s.$$.ptrType.registeredClass,u=ps(s.$$.ptr,c,this.registeredClass);return u}function Sf(i,s){var c;if(s===null)return this.isReference&&mt(`null is not a valid ${this.name}`),this.isSmartPointer?(c=this.rawConstructor(),i!==null&&i.push(this.rawDestructor,c),c):0;(!s||!s.$$)&&mt(`Cannot pass "${Ci(s)}" as a ${this.name}`),s.$$.ptr||mt(`Cannot pass deleted object as a pointer of type ${this.name}`),!this.isConst&&s.$$.ptrType.isConst&&mt(`Cannot convert argument of type ${s.$$.smartPtrType?s.$$.smartPtrType.name:s.$$.ptrType.name} to parameter type ${this.name}`);var u=s.$$.ptrType.registeredClass;if(c=ps(s.$$.ptr,u,this.registeredClass),this.isSmartPointer)switch(s.$$.smartPtr===void 0&&mt("Passing raw pointer to smart pointer is illegal"),this.sharingPolicy){case 0:s.$$.smartPtrType===this?c=s.$$.smartPtr:mt(`Cannot convert argument of type ${s.$$.smartPtrType?s.$$.smartPtrType.name:s.$$.ptrType.name} to parameter type ${this.name}`);break;case 1:c=s.$$.smartPtr;break;case 2:if(s.$$.smartPtrType===this)c=s.$$.smartPtr;else{var d=s.clone();c=this.rawShare(c,Xt.toHandle(()=>d.delete())),i!==null&&i.push(this.rawDestructor,c)}break;default:mt("Unsupporting sharing policy")}return c}function bf(i,s){if(s===null)return this.isReference&&mt(`null is not a valid ${this.name}`),0;s.$$||mt(`Cannot pass "${Ci(s)}" as a ${this.name}`),s.$$.ptr||mt(`Cannot pass deleted object as a pointer of type ${this.name}`),s.$$.ptrType.isConst&&mt(`Cannot convert argument of type ${s.$$.ptrType.name} to parameter type ${this.name}`);var c=s.$$.ptrType.registeredClass,u=ps(s.$$.ptr,c,this.registeredClass);return u}function ms(i){return this.fromWireType(ye[i>>2])}var Ef=()=>{Object.assign(_s.prototype,{getPointee(i){return this.rawGetPointee&&(i=this.rawGetPointee(i)),i},destructor(i){this.rawDestructor?.(i)},argPackAdvance:kn,readValueFromPointer:ms,fromWireType:Lc})};function _s(i,s,c,u,d,x,M,T,I,X,q){this.name=i,this.registeredClass=s,this.isReference=c,this.isConst=u,this.isSmartPointer=d,this.pointeeType=x,this.sharingPolicy=M,this.rawGetPointee=T,this.rawConstructor=I,this.rawShare=X,this.rawDestructor=q,!d&&s.baseClass===void 0?u?(this.toWireType=yf,this.destructorFunction=null):(this.toWireType=bf,this.destructorFunction=null):this.toWireType=Sf}var Fc=(i,s,c)=>{t.hasOwnProperty(i)||us("Replacing nonexistent public symbol"),t[i].overloadTable!==void 0&&c!==void 0?t[i].overloadTable[c]=s:(t[i]=s,t[i].argCount=c)},Ic=[],xs,xe=i=>{var s=Ic[i];return s||(Ic[i]=s=xs.get(i)),A(xs.get(i)==s,"JavaScript-side Wasm function table mirror is out of date!"),s},Bn=(i,s,c=!1)=>{A(!c,"Async bindings are only supported with JSPI."),i=Wt(i);function u(){var x=xe(s);return x}var d=u();return typeof d!="function"&&mt(`unknown function pointer with signature ${i}: ${s}`),d};class Mf extends Error{}var Uc=i=>{var s=el(i),c=Wt(s);return Vn(s),c},Pi=(i,s)=>{var c=[],u={};function d(x){if(!u[x]&&!Ri[x]){if(ls[x]){ls[x].forEach(d);return}c.push(x),u[x]=!0}}throw s.forEach(d),new Mf(`${i}: `+c.map(Uc).join([", "]))},Rn=(i,s,c)=>{i.forEach(T=>ls[T]=s);function u(T){var I=c(T);I.length!==i.length&&us("Mismatched type converter count");for(var X=0;X<i.length;++X)Sn(i[X],I[X])}var d=new Array(s.length),x=[],M=0;s.forEach((T,I)=>{Ri.hasOwnProperty(T)?d[I]=Ri[T]:(x.push(T),Ki.hasOwnProperty(T)||(Ki[T]=[]),Ki[T].push(()=>{d[I]=Ri[T],++M,M===x.length&&u(d)}))}),x.length===0&&u(d)},Tf=(i,s,c,u,d,x,M,T,I,X,q,ae,ue)=>{q=Wt(q),x=Bn(d,x),T&&=Bn(M,T),X&&=Bn(I,X),ue=Bn(ae,ue);var oe=gf(q);Ea(oe,function(){Pi(`Cannot construct ${q} due to unbound types`,[u])}),Rn([i,s,c],u?[u]:[],pe=>{pe=pe[0];var Xe,ht;u?(Xe=pe.registeredClass,ht=Xe.instancePrototype):ht=ds.prototype;var rt=hs(q,function(...Lt){if(Object.getPrototypeOf(this)!==Ut)throw new Ur(`Use 'new' to construct ${q}`);if(gt.constructor_body===void 0)throw new Ur(`${q} has no accessible constructor`);var Fi=gt.constructor_body[Lt.length];if(Fi===void 0)throw new Ur(`Tried to invoke ctor of ${q} with invalid number of parameters (${Lt.length}) - expected (${Object.keys(gt.constructor_body).toString()}) parameters instead!`);return Fi.apply(this,Lt)}),Ut=Object.create(ht,{constructor:{value:rt}});rt.prototype=Ut;var gt=new vf(q,rt,Ut,ue,Xe,x,T,X);gt.baseClass&&(gt.baseClass.__derivedClasses??=[],gt.baseClass.__derivedClasses.push(gt));var rn=new _s(q,gt,!0,!1,!1),$t=new _s(q+"*",gt,!1,!1,!1),cn=new _s(q+" const*",gt,!1,!0,!1);return Dc[i]={pointerType:$t,constPointerType:cn},Fc(oe,rt),[rn,$t,cn]})},Ma=i=>{for(;i.length;){var s=i.pop(),c=i.pop();c(s)}};function Nc(i){for(var s=1;s<i.length;++s)if(i[s]!==null&&i[s].destructorFunction===void 0)return!0;return!1}function wf(i,s,c,u,d){if(i<s||i>c){var x=s==c?s:`${s} to ${c}`;d(`function ${u} called with ${i} arguments, expected ${x}`)}}function Af(i,s,c,u){var d=Nc(i),x=i.length-2,M=[],T=["fn"];s&&T.push("thisWired");for(var I=0;I<x;++I)M.push(`arg${I}`),T.push(`arg${I}Wired`);M=M.join(","),T=T.join(",");var X=`return function (${M}) {
`;X+=`checkArgCount(arguments.length, minArgs, maxArgs, humanName, throwBindingError);
`,d&&(X+=`var destructors = [];
`);var q=d?"destructors":"null",ae=["humanName","throwBindingError","invoker","fn","runDestructors","retType","classParam"];s&&(X+=`var thisWired = classParam['toWireType'](${q}, this);
`);for(var I=0;I<x;++I)X+=`var arg${I}Wired = argType${I}['toWireType'](${q}, arg${I});
`,ae.push(`argType${I}`);if(X+=(c||u?"var rv = ":"")+`invoker(${T});
`,d)X+=`runDestructors(destructors);
`;else for(var I=s?1:2;I<i.length;++I){var ue=I===1?"thisWired":"arg"+(I-2)+"Wired";i[I].destructorFunction!==null&&(X+=`${ue}_dtor(${ue});
`,ae.push(`${ue}_dtor`))}return c&&(X+=`var ret = retType['fromWireType'](rv);
return ret;
`),X+=`}
`,ae.push("checkArgCount","minArgs","maxArgs"),X=`if (arguments.length !== ${ae.length}){ throw new Error(humanName + "Expected ${ae.length} closure arguments " + arguments.length + " given."); }
${X}`,[ae,X]}function Rf(i){for(var s=i.length-2,c=i.length-1;c>=2&&i[c].optional;--c)s--;return s}function gs(i,s,c,u,d,x){var M=s.length;M<2&&mt("argTypes array size mismatch! Must at least get return value and 'this' types!"),A(!x,"Async bindings are only supported with JSPI.");for(var T=s[1]!==null&&c!==null,I=Nc(s),X=s[0].name!=="void",q=M-2,ae=Rf(s),ue=[i,mt,u,d,Ma,s[0],s[1]],oe=0;oe<M-2;++oe)ue.push(s[oe+2]);if(!I)for(var oe=T?1:2;oe<s.length;++oe)s[oe].destructorFunction!==null&&ue.push(s[oe].destructorFunction);ue.push(wf,ae,q);let[pe,Xe]=Af(s,T,X,x);var ht=new Function(...pe,Xe)(...ue);return hs(i,ht)}var vs=(i,s)=>{for(var c=[],u=0;u<i;u++)c.push(ye[s+u*4>>2]);return c},Ta=i=>{i=i.trim();const s=i.indexOf("(");return s===-1?i:(A(i.endsWith(")"),"Parentheses for argument names should match."),i.slice(0,s))},Cf=(i,s,c,u,d,x,M,T,I)=>{var X=vs(c,u);s=Wt(s),s=Ta(s),x=Bn(d,x,T),Rn([],[i],q=>{q=q[0];var ae=`${q.name}.${s}`;function ue(){Pi(`Cannot call ${ae} due to unbound types`,X)}s.startsWith("@@")&&(s=Symbol[s.substring(2)]);var oe=q.registeredClass.constructor;return oe[s]===void 0?(ue.argCount=c-1,oe[s]=ue):(ba(oe,s,ae),oe[s].overloadTable[c-1]=ue),Rn([],X,pe=>{var Xe=[pe[0],null].concat(pe.slice(1)),ht=gs(ae,Xe,null,x,M,T);if(oe[s].overloadTable===void 0?(ht.argCount=c-1,oe[s]=ht):oe[s].overloadTable[c-1]=ht,q.registeredClass.__derivedClasses)for(const rt of q.registeredClass.__derivedClasses)rt.constructor.hasOwnProperty(s)||(rt.constructor[s]=ht);return[]}),[]})},Pf=(i,s,c,u,d,x)=>{A(s>0);var M=vs(s,c);d=Bn(u,d),Rn([],[i],T=>{T=T[0];var I=`constructor ${T.name}`;if(T.registeredClass.constructor_body===void 0&&(T.registeredClass.constructor_body=[]),T.registeredClass.constructor_body[s-1]!==void 0)throw new Ur(`Cannot register multiple constructors with identical number of parameters (${s-1}) for class '${T.name}'! Overload resolution is currently only performed using the parameter count, not actual type info!`);return T.registeredClass.constructor_body[s-1]=()=>{Pi(`Cannot construct ${T.name} due to unbound types`,M)},Rn([],M,X=>(X.splice(1,0,null),T.registeredClass.constructor_body[s-1]=gs(I,X,null,d,x),[])),[]})},Df=(i,s,c,u,d,x,M,T,I,X)=>{var q=vs(c,u);s=Wt(s),s=Ta(s),x=Bn(d,x,I),Rn([],[i],ae=>{ae=ae[0];var ue=`${ae.name}.${s}`;s.startsWith("@@")&&(s=Symbol[s.substring(2)]),T&&ae.registeredClass.pureVirtualFunctions.push(s);function oe(){Pi(`Cannot call ${ue} due to unbound types`,q)}var pe=ae.registeredClass.instancePrototype,Xe=pe[s];return Xe===void 0||Xe.overloadTable===void 0&&Xe.className!==ae.name&&Xe.argCount===c-2?(oe.argCount=c-2,oe.className=ae.name,pe[s]=oe):(ba(pe,s,ue),pe[s].overloadTable[c-2]=oe),Rn([],q,ht=>{var rt=gs(ue,ht,ae,x,M,I);return pe[s].overloadTable===void 0?(rt.argCount=c-2,pe[s]=rt):pe[s].overloadTable[c-2]=rt,[]}),[]})},Oc=(i,s,c)=>(i instanceof Object||mt(`${c} with invalid "this": ${i}`),i instanceof s.registeredClass.constructor||mt(`${c} incompatible with "this" of type ${i.constructor.name}`),i.$$.ptr||mt(`cannot call emscripten binding method ${c} on deleted object`),ps(i.$$.ptr,i.$$.ptrType.registeredClass,s.registeredClass)),Lf=(i,s,c,u,d,x,M,T,I,X)=>{s=Wt(s),d=Bn(u,d),Rn([],[i],q=>{q=q[0];var ae=`${q.name}.${s}`,ue={get(){Pi(`Cannot access ${ae} due to unbound types`,[c,M])},enumerable:!0,configurable:!0};return I?ue.set=()=>Pi(`Cannot access ${ae} due to unbound types`,[c,M]):ue.set=oe=>mt(ae+" is a read-only property"),Object.defineProperty(q.registeredClass.instancePrototype,s,ue),Rn([],I?[c,M]:[c],oe=>{var pe=oe[0],Xe={get(){var rt=Oc(this,q,ae+" getter");return pe.fromWireType(d(x,rt))},enumerable:!0};if(I){I=Bn(T,I);var ht=oe[1];Xe.set=function(rt){var Ut=Oc(this,q,ae+" setter"),gt=[];I(X,Ut,ht.toWireType(gt,rt)),Ma(gt)}}return Object.defineProperty(q.registeredClass.instancePrototype,s,Xe),[]}),[]})},Ff=(i,s,c)=>{i=Wt(i),Rn([],[s],u=>(u=u[0],t[i]=u.fromWireType(c),[]))},kc=[],zn=[0,1,,1,null,1,!0,1,!1,1],wa=i=>{i>9&&--zn[i+1]===0&&(A(zn[i]!==void 0,"Decref for unallocated handle."),zn[i]=void 0,kc.push(i))},Xt={toValue:i=>(i||mt(`Cannot use deleted val. handle = ${i}`),A(i===2||zn[i]!==void 0&&i%2===0,`invalid handle: ${i}`),zn[i]),toHandle:i=>{switch(i){case void 0:return 2;case null:return 4;case!0:return 6;case!1:return 8;default:{const s=kc.pop()||zn.length;return zn[s]=i,zn[s+1]=1,s}}}},Bc={name:"emscripten::val",fromWireType:i=>{var s=Xt.toValue(i);return wa(i),s},toWireType:(i,s)=>Xt.toHandle(s),argPackAdvance:kn,readValueFromPointer:ms,destructorFunction:null},zc=i=>Sn(i,Bc),If=(i,s,c)=>{switch(s){case 1:return c?function(u){return this.fromWireType(Je[u])}:function(u){return this.fromWireType(at[u])};case 2:return c?function(u){return this.fromWireType(ce[u>>1])}:function(u){return this.fromWireType(de[u>>1])};case 4:return c?function(u){return this.fromWireType(ie[u>>2])}:function(u){return this.fromWireType(ye[u>>2])};default:throw new TypeError(`invalid integer width (${s}): ${i}`)}},Uf=(i,s,c,u)=>{s=Wt(s);function d(){}d.values={},Sn(i,{name:s,constructor:d,fromWireType:function(x){return this.constructor.values[x]},toWireType:(x,M)=>M.value,argPackAdvance:kn,readValueFromPointer:If(s,c,u),destructorFunction:null}),Ea(s,d)},ys=(i,s)=>{var c=Ri[i];return c===void 0&&mt(`${s} has unknown type ${Uc(i)}`),c},Nf=(i,s,c)=>{var u=ys(i,"enum");s=Wt(s);var d=u.constructor,x=Object.create(u.constructor.prototype,{value:{value:c},constructor:{value:hs(`${u.name}_${s}`,function(){})}});d.values[c]=x,d[s]=x},Of=(i,s)=>{switch(s){case 4:return function(c){return this.fromWireType(Be[c>>2])};case 8:return function(c){return this.fromWireType(nt[c>>3])};default:throw new TypeError(`invalid float width (${s}): ${i}`)}},kf=(i,s,c)=>{s=Wt(s),Sn(i,{name:s,fromWireType:u=>u,toWireType:(u,d)=>{if(typeof d!="number"&&typeof d!="boolean")throw new TypeError(`Cannot convert ${Ci(d)} to ${this.name}`);return d},argPackAdvance:kn,readValueFromPointer:Of(s,c),destructorFunction:null})},Bf=(i,s,c,u,d,x,M,T)=>{var I=vs(s,c);i=Wt(i),i=Ta(i),d=Bn(u,d,M),Ea(i,function(){Pi(`Cannot call ${i} due to unbound types`,I)},s-1),Rn([],I,X=>{var q=[X[0],null].concat(X.slice(1));return Fc(i,gs(i,q,null,d,x,M),s-1),[]})},zf=(i,s,c,u,d)=>{s=Wt(s);const x=u===0;let M=I=>I;if(x){var T=32-8*c;M=I=>I<<T>>>T,d=M(d)}Sn(i,{name:s,fromWireType:M,toWireType:(I,X)=>{if(typeof X!="number"&&typeof X!="boolean")throw new TypeError(`Cannot convert "${Ci(X)}" to ${s}`);return Ac(s,X,u,d),X},argPackAdvance:kn,readValueFromPointer:wc(s,c,u!==0),destructorFunction:null})},Vf=(i,s,c)=>{var u=[Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array,BigInt64Array,BigUint64Array],d=u[s];function x(M){var T=ye[M>>2],I=ye[M+4>>2];return new d(Je.buffer,I,T)}c=Wt(c),Sn(i,{name:c,fromWireType:x,argPackAdvance:kn,readValueFromPointer:x},{ignoreDuplicateRegistrations:!0})},Gf=Object.assign({optional:!0},Bc),Hf=(i,s)=>{Sn(i,Gf)},Di=(i,s,c)=>(A(typeof c=="number","stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!"),j(i,at,s,c)),Wf=(i,s)=>{s=Wt(s),Sn(i,{name:s,fromWireType(c){for(var u=ye[c>>2],d=c+4,x,M,T=d,M=0;M<=u;++M){var I=d+M;if(M==u||at[I]==0){var X=I-T,q=ut(T,X);x===void 0?x=q:(x+="\0",x+=q),T=I+1}}return Vn(c),x},toWireType(c,u){u instanceof ArrayBuffer&&(u=new Uint8Array(u));var d,x=typeof u=="string";x||ArrayBuffer.isView(u)&&u.BYTES_PER_ELEMENT==1||mt("Cannot pass non-string to std::string"),x?d=Q(u):d=u.length;var M=Ca(4+d+1),T=M+4;return ye[M>>2]=d,x?Di(u,T,d+1):at.set(u,T),c!==null&&c.push(Vn,M),M},argPackAdvance:kn,readValueFromPointer:ms,destructorFunction(c){Vn(c)}})},Vc=typeof TextDecoder<"u"?new TextDecoder("utf-16le"):void 0,Xf=(i,s)=>{A(i%2==0,"Pointer passed to UTF16ToString must be aligned to two bytes!");for(var c=i>>1,u=c+s/2,d=c;!(d>=u)&&de[d];)++d;if(d-c>16&&Vc)return Vc.decode(de.subarray(c,d));for(var x="",M=c;!(M>=u);++M){var T=de[M];if(T==0)break;x+=String.fromCharCode(T)}return x},$f=(i,s,c)=>{if(A(s%2==0,"Pointer passed to stringToUTF16 must be aligned to two bytes!"),A(typeof c=="number","stringToUTF16(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!"),c??=2147483647,c<2)return 0;c-=2;for(var u=s,d=c<i.length*2?c/2:i.length,x=0;x<d;++x){var M=i.charCodeAt(x);ce[s>>1]=M,s+=2}return ce[s>>1]=0,s-u},jf=i=>i.length*2,Yf=(i,s)=>{A(i%4==0,"Pointer passed to UTF32ToString must be aligned to four bytes!");for(var c="",u=0;!(u>=s/4);u++){var d=ie[i+u*4>>2];if(!d)break;c+=String.fromCodePoint(d)}return c},qf=(i,s,c)=>{if(A(s%4==0,"Pointer passed to stringToUTF32 must be aligned to four bytes!"),A(typeof c=="number","stringToUTF32(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!"),c??=2147483647,c<4)return 0;for(var u=s,d=u+c-4,x=0;x<i.length;++x){var M=i.codePointAt(x);if(M>65535&&x++,ie[s>>2]=M,s+=4,s+4>d)break}return ie[s>>2]=0,s-u},Kf=i=>{for(var s=0,c=0;c<i.length;++c){var u=i.codePointAt(c);u>65535&&c++,s+=4}return s},Zf=(i,s,c)=>{c=Wt(c);var u,d,x,M;s===2?(u=Xf,d=$f,M=jf,x=T=>de[T>>1]):s===4&&(u=Yf,d=qf,M=Kf,x=T=>ye[T>>2]),Sn(i,{name:c,fromWireType:T=>{for(var I=ye[T>>2],X,q=T+4,ae=0;ae<=I;++ae){var ue=T+4+ae*s;if(ae==I||x(ue)==0){var oe=ue-q,pe=u(q,oe);X===void 0?X=pe:(X+="\0",X+=pe),q=ue+s}}return Vn(T),X},toWireType:(T,I)=>{typeof I!="string"&&mt(`Cannot pass non-string to C++ string type ${c}`);var X=M(I),q=Ca(4+X+s);return ye[q>>2]=X/s,d(I,q+4,X+s),T!==null&&T.push(Vn,q),q},argPackAdvance:kn,readValueFromPointer:ms,destructorFunction(T){Vn(T)}})},Jf=(i,s)=>{zc(i)},Qf=(i,s)=>{s=Wt(s),Sn(i,{isVoid:!0,name:s,argPackAdvance:0,fromWireType:()=>{},toWireType:(c,u)=>{}})},ed=()=>{throw new G},Gc=(i,s,c)=>{var u=[],d=i.toWireType(u,c);return u.length&&(ye[s>>2]=Xt.toHandle(u)),d},td=(i,s,c)=>(i=Xt.toValue(i),s=ys(s,"emval::as"),Gc(s,c,i)),Ss=[],nd=(i,s,c,u)=>(i=Ss[i],s=Xt.toValue(s),i(null,s,c,u)),id={},Aa=i=>{var s=id[i];return s===void 0?Wt(i):s},rd=(i,s,c,u,d)=>(i=Ss[i],s=Xt.toValue(s),c=Aa(c),i(s,s[c],u,d)),Hc=()=>globalThis,sd=i=>i===0?Xt.toHandle(Hc()):(i=Aa(i),Xt.toHandle(Hc()[i])),ad=i=>{var s=Ss.length;return Ss.push(i),s},od=(i,s)=>{for(var c=new Array(i),u=0;u<i;++u)c[u]=ys(ye[s+u*4>>2],`parameter ${u}`);return c},cd=(i,s,c)=>{var u=od(i,s),d=u.shift();i--;var x=`return function (obj, func, destructorsRef, args) {
`,M=0,T=[];c===0&&T.push("obj");for(var I=["retType"],X=[d],q=0;q<i;++q)T.push(`arg${q}`),I.push(`argType${q}`),X.push(u[q]),x+=`  var arg${q} = argType${q}.readValueFromPointer(args${M?"+"+M:""});
`,M+=u[q].argPackAdvance;var ae=c===1?"new func":"func.call";x+=`  var rv = ${ae}(${T.join(", ")});
`,d.isVoid||(I.push("emval_returnValue"),X.push(Gc),x+=`  return emval_returnValue(retType, destructorsRef, rv);
`),x+=`};
`;var ue=new Function(...I,x)(...X),oe=`methodCaller<(${u.map(pe=>pe.name).join(", ")}) => ${d.name}>`;return ad(hs(oe,ue))},ld=(i,s)=>(i=Xt.toValue(i),s=Xt.toValue(s),Xt.toHandle(i[s])),ud=i=>{i>9&&(zn[i+1]+=1)},fd=i=>(i=Xt.toValue(i),typeof i=="number"),dd=i=>(i=Xt.toValue(i),typeof i=="string"),hd=()=>Xt.toHandle([]),pd=i=>Xt.toHandle(Aa(i)),md=i=>{var s=Xt.toValue(i);Ma(s),wa(i)},_d=(i,s)=>{i=ys(i,"_emval_take_value");var c=i.readValueFromPointer(s);return Xt.toHandle(c)},xd=i=>{throw i=Xt.toValue(i),i},gd=i=>i%4===0&&(i%100!==0||i%400===0),vd=[0,31,60,91,121,152,182,213,244,274,305,335],yd=[0,31,59,90,120,151,181,212,243,273,304,334],Wc=i=>{var s=gd(i.getFullYear()),c=s?vd:yd,u=c[i.getMonth()]+i.getDate()-1;return u},Sd=9007199254740992,bd=-9007199254740992,Xc=i=>i<bd||i>Sd?NaN:Number(i);function Ed(i,s){i=Xc(i);var c=new Date(i*1e3);ie[s>>2]=c.getSeconds(),ie[s+4>>2]=c.getMinutes(),ie[s+8>>2]=c.getHours(),ie[s+12>>2]=c.getDate(),ie[s+16>>2]=c.getMonth(),ie[s+20>>2]=c.getFullYear()-1900,ie[s+24>>2]=c.getDay();var u=Wc(c)|0;ie[s+28>>2]=u,ie[s+36>>2]=-(c.getTimezoneOffset()*60);var d=new Date(c.getFullYear(),0,1),x=new Date(c.getFullYear(),6,1).getTimezoneOffset(),M=d.getTimezoneOffset(),T=(x!=M&&c.getTimezoneOffset()==Math.min(M,x))|0;ie[s+32>>2]=T}var Md=function(i){var s=(()=>{var c=new Date(ie[i+20>>2]+1900,ie[i+16>>2],ie[i+12>>2],ie[i+8>>2],ie[i+4>>2],ie[i>>2],0),u=ie[i+32>>2],d=c.getTimezoneOffset(),x=new Date(c.getFullYear(),0,1),M=new Date(c.getFullYear(),6,1).getTimezoneOffset(),T=x.getTimezoneOffset(),I=Math.min(T,M);if(u<0)ie[i+32>>2]=+(M!=T&&I==d);else if(u>0!=(I==d)){var X=Math.max(T,M),q=u>0?I:X;c.setTime(c.getTime()+(q-d)*6e4)}ie[i+24>>2]=c.getDay();var ae=Wc(c)|0;ie[i+28>>2]=ae,ie[i>>2]=c.getSeconds(),ie[i+4>>2]=c.getMinutes(),ie[i+8>>2]=c.getHours(),ie[i+12>>2]=c.getDate(),ie[i+16>>2]=c.getMonth(),ie[i+20>>2]=c.getYear();var ue=c.getTime();return isNaN(ue)?-1:ue/1e3})();return BigInt(s)},Td=(i,s,c,u)=>{var d=new Date().getFullYear(),x=new Date(d,0,1),M=new Date(d,6,1),T=x.getTimezoneOffset(),I=M.getTimezoneOffset(),X=Math.max(T,I);ye[i>>2]=X*60,ie[s>>2]=+(T!=I);var q=oe=>{var pe=oe>=0?"-":"+",Xe=Math.abs(oe),ht=String(Math.floor(Xe/60)).padStart(2,"0"),rt=String(Xe%60).padStart(2,"0");return`UTC${pe}${ht}${rt}`},ae=q(T),ue=q(I);A(ae),A(ue),A(Q(ae)<=16,`timezone name truncated to fit in TZNAME_MAX (${ae})`),A(Q(ue)<=16,`timezone name truncated to fit in TZNAME_MAX (${ue})`),I<T?(Di(ae,c,17),Di(ue,u,17)):(Di(ae,u,17),Di(ue,c,17))},$c=()=>performance.now(),jc=()=>Date.now(),wd=i=>i>=0&&i<=3;function Ad(i,s,c){if(!wd(i))return 28;var u;i===0?u=jc():u=$c();var d=Math.round(u*1e3*1e3);return Pt[c>>3]=BigInt(d),0}var bs=[],Rd=(i,s)=>{A(Array.isArray(bs)),A(s%16==0),bs.length=0;for(var c;c=at[i++];){var u=String.fromCharCode(c),d=["d","f","i","p"];d.push("j"),A(d.includes(u),`Invalid character ${c}("${u}") in readEmAsmArgs! Use only [${d}], and do not specify "v" for void return argument.`);var x=c!=105;x&=c!=112,s+=x&&s%8?4:0,bs.push(c==112?ye[s>>2]:c==106?Pt[s>>3]:c==105?ie[s>>2]:nt[s>>3]),s+=x?8:4}return bs},Cd=(i,s,c)=>{var u=Rd(s,c);return A(Qc.hasOwnProperty(i),`No EM_ASM constant found at address ${i}.  The loaded WebAssembly file is likely out of sync with the generated JavaScript.`),Qc[i](...u)},Pd=(i,s,c)=>Cd(i,s,c),Yc=()=>2147483648,Dd=()=>Yc(),Ld=(i,s)=>(A(s,"alignment argument is required"),Math.ceil(i/s)*s),Fd=i=>{var s=_t.buffer,c=(i-s.byteLength+65535)/65536|0;try{return _t.grow(c),O(),1}catch(u){L(`growMemory: Attempted to grow heap from ${s.byteLength} bytes to ${i} bytes, but got error: ${u}`)}},Id=i=>{var s=at.length;i>>>=0,A(i>s);var c=Yc();if(i>c)return L(`Cannot enlarge memory, requested ${i} bytes, but the limit is ${c} bytes!`),!1;for(var u=1;u<=4;u*=2){var d=s*(1+.2/u);d=Math.min(d,i+100663296);var x=Math.min(c,Ld(Math.max(i,d),65536)),M=Fd(x);if(M)return!0}return L(`Failed to grow the heap from ${s} bytes to ${x} bytes, not enough memory!`),!1},Ra={},Ud=()=>p||"./this.program",Or=()=>{if(!Or.strings){var i=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8",s={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:i,_:Ud()};for(var c in Ra)Ra[c]===void 0?delete s[c]:s[c]=Ra[c];var u=[];for(var c in s)u.push(`${c}=${s[c]}`);Or.strings=u}return Or.strings},Nd=(i,s)=>{var c=0,u=0;for(var d of Or()){var x=s+c;ye[i+u>>2]=x,c+=Di(d,x,1/0)+1,u+=4}return 0},Od=(i,s)=>{var c=Or();ye[i>>2]=c.length;var u=0;for(var d of c)u+=Q(d)+1;return ye[s>>2]=u,0},qc=0,Kc=()=>Te||qc>0,kd=i=>{Kc()||(t.onExit?.(i),z=!0),h(i,new je(i))},Bd=(i,s)=>{if(Kp(),Kc()&&!s){var c=`program exited (with status: ${i}), but keepRuntimeAlive() is set (counter=${qc}) due to an async operation, so halting execution but not exiting the runtime or preventing further async execution (you can use emscripten_force_exit, if you want to force a true shutdown)`;ot?.(c),L(c)}kd(i)},zd=Bd;function Vd(i){try{var s=Et.getStreamFromFD(i);return S.close(s),0}catch(c){if(typeof S>"u"||c.name!=="ErrnoError")throw c;return c.errno}}var Gd=(i,s,c,u)=>{for(var d=0,x=0;x<c;x++){var M=ye[s>>2],T=ye[s+4>>2];s+=8;var I=S.read(i,Je,M,T,u);if(I<0)return-1;if(d+=I,I<T)break}return d};function Hd(i,s,c,u){try{var d=Et.getStreamFromFD(i),x=Gd(d,s,c);return ye[u>>2]=x,0}catch(M){if(typeof S>"u"||M.name!=="ErrnoError")throw M;return M.errno}}function Wd(i,s,c,u){s=Xc(s);try{if(isNaN(s))return 61;var d=Et.getStreamFromFD(i);return S.llseek(d,s,c),Pt[u>>3]=BigInt(d.position),d.getdents&&s===0&&c===0&&(d.getdents=null),0}catch(x){if(typeof S>"u"||x.name!=="ErrnoError")throw x;return x.errno}}var Xd=(i,s,c,u)=>{for(var d=0,x=0;x<c;x++){var M=ye[s>>2],T=ye[s+4>>2];s+=8;var I=S.write(i,Je,M,T,u);if(I<0)return-1;if(d+=I,I<T)break}return d};function $d(i,s,c,u){try{var d=Et.getStreamFromFD(i),x=Xd(d,s,c);return ye[u>>2]=x,0}catch(M){if(typeof S>"u"||M.name!=="ErrnoError")throw M;return M.errno}}var jd=i=>i,Yd=i=>{var s=t["_"+i];return A(s,"Cannot call unknown function "+i+", make sure it is exported"),s},qd=(i,s)=>{A(i.length>=0,"writeArrayToMemory array must have a length (should be an array or typed array)"),Je.set(i,s)},Es=i=>sl(i),Kd=i=>{var s=Q(i)+1,c=Es(s);return Di(i,c,s),c},Zc=(i,s,c,u,d)=>{var x={string:pe=>{var Xe=0;return pe!=null&&pe!==0&&(Xe=Kd(pe)),Xe},array:pe=>{var Xe=Es(pe.length);return qd(pe,Xe),Xe}};function M(pe){return s==="string"?ut(pe):s==="boolean"?!!pe:pe}var T=Yd(i),I=[],X=0;if(A(s!=="array",'Return type should not be "array".'),u)for(var q=0;q<u.length;q++){var ae=x[c[q]];ae?(X===0&&(X=B()),I[q]=ae(u[q])):I[q]=u[q]}var ue=T(...I);function oe(pe){return X!==0&&H(X),M(pe)}return ue=oe(ue),ue},Zd=(i,s,c,u)=>(...d)=>Zc(i,s,c,d),Jd=(...i)=>S.createPath(...i),Qd=(...i)=>S.unlink(...i),eh=(...i)=>S.createLazyFile(...i),th=(...i)=>S.createDevice(...i),nh=i=>Ms(i),ih=i=>La(i),rh=i=>{var s=B(),c=Es(4),u=Es(4);ol(i,c,u);var d=ye[c>>2],x=ye[u>>2],M=ut(d);Vn(d);var T;return x&&(T=ut(x),Vn(x)),H(s),[M,T]},Jc=i=>rh(i);S.createPreloadedFile=Nt,S.staticInit(),mf(),Ef(),A(zn.length===10),t.noExitRuntime&&(Te=t.noExitRuntime),t.preloadPlugins&&(bt=t.preloadPlugins),t.print&&(U=t.print),t.printErr&&(L=t.printErr),t.wasmBinary&&(N=t.wasmBinary),oh(),t.arguments&&t.arguments,t.thisProgram&&(p=t.thisProgram),A(typeof t.memoryInitializerPrefixURL>"u","Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead"),A(typeof t.pthreadMainPrefixURL>"u","Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead"),A(typeof t.cdInitializerPrefixURL>"u","Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead"),A(typeof t.filePackagePrefixURL>"u","Module.filePackagePrefixURL option was removed, use Module.locateFile instead"),A(typeof t.read>"u","Module.read option was removed"),A(typeof t.readAsync>"u","Module.readAsync option was removed (modify readAsync in JS)"),A(typeof t.readBinary>"u","Module.readBinary option was removed (modify readBinary in JS)"),A(typeof t.setWindowTitle>"u","Module.setWindowTitle option was removed (modify emscripten_set_window_title in JS)"),A(typeof t.TOTAL_MEMORY>"u","Module.TOTAL_MEMORY has been renamed Module.INITIAL_MEMORY"),A(typeof t.ENVIRONMENT>"u","Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -sENVIRONMENT=web or -sENVIRONMENT=node)"),A(typeof t.STACK_SIZE>"u","STACK_SIZE can no longer be set at runtime.  Use -sSTACK_SIZE at link time"),A(typeof t.wasmMemory>"u","Use of `wasmMemory` detected.  Use -sIMPORTED_MEMORY to define wasmMemory externally"),A(typeof t.INITIAL_MEMORY>"u","Detected runtime INITIAL_MEMORY setting.  Use -sIMPORTED_MEMORY to define wasmMemory dynamically"),t.addRunDependency=F,t.removeRunDependency=w,t.ccall=Zc,t.cwrap=Zd,t.FS_createPreloadedFile=Nt,t.FS_unlink=Qd,t.FS_createPath=Jd,t.FS_createDevice=th,t.FS=S,t.FS_createDataFile=ze,t.FS_createLazyFile=eh,t.MEMFS=ge;var sh=["writeI53ToI64","writeI53ToI64Clamped","writeI53ToI64Signaling","writeI53ToU64Clamped","writeI53ToU64Signaling","readI53FromI64","readI53FromU64","convertI32PairToI53","convertI32PairToI53Checked","convertU32PairToI53","getTempRet0","zeroMemory","withStackSave","inetPton4","inetNtop4","inetPton6","inetNtop6","readSockaddr","writeSockaddr","emscriptenLog","runMainThreadEmAsm","jstoi_q","autoResumeAudioContext","getDynCaller","dynCall","handleException","runtimeKeepalivePush","runtimeKeepalivePop","callUserCallback","maybeExit","asmjsMangle","HandleAllocator","getNativeTypeSize","addOnInit","addOnPostCtor","addOnPreMain","addOnExit","STACK_SIZE","STACK_ALIGN","POINTER_SIZE","ASSERTIONS","uleb128Encode","sigToWasmTypes","generateFuncType","convertJsFunctionToWasm","getEmptyTableSlot","updateTableMap","getFunctionAddress","addFunction","removeFunction","reallyNegative","unSign","strLen","reSign","formatString","intArrayToString","stringToAscii","stringToNewUTF8","registerKeyEventCallback","maybeCStringToJsString","findEventTarget","getBoundingClientRect","fillMouseEventData","registerMouseEventCallback","registerWheelEventCallback","registerUiEventCallback","registerFocusEventCallback","fillDeviceOrientationEventData","registerDeviceOrientationEventCallback","fillDeviceMotionEventData","registerDeviceMotionEventCallback","screenOrientation","fillOrientationChangeEventData","registerOrientationChangeEventCallback","fillFullscreenChangeEventData","registerFullscreenChangeEventCallback","JSEvents_requestFullscreen","JSEvents_resizeCanvasForFullscreen","registerRestoreOldStyle","hideEverythingExceptGivenElement","restoreHiddenElements","setLetterbox","softFullscreenResizeWebGLRenderTarget","doRequestFullscreen","fillPointerlockChangeEventData","registerPointerlockChangeEventCallback","registerPointerlockErrorEventCallback","requestPointerLock","fillVisibilityChangeEventData","registerVisibilityChangeEventCallback","registerTouchEventCallback","fillGamepadEventData","registerGamepadEventCallback","registerBeforeUnloadEventCallback","fillBatteryEventData","battery","registerBatteryEventCallback","setCanvasElementSize","getCanvasElementSize","jsStackTrace","getCallstack","convertPCtoSourceLocation","wasiRightsToMuslOFlags","wasiOFlagsToMuslOFlags","safeSetTimeout","setImmediateWrapped","safeRequestAnimationFrame","clearImmediateWrapped","registerPostMainLoop","registerPreMainLoop","getPromise","makePromise","idsToPromises","makePromiseCallback","Browser_asyncPrepareDataCounter","arraySum","addDays","getSocketFromFD","getSocketAddress","FS_mkdirTree","_setNetworkCallback","heapObjectForWebGLType","toTypedArrayIndex","webgl_enable_ANGLE_instanced_arrays","webgl_enable_OES_vertex_array_object","webgl_enable_WEBGL_draw_buffers","webgl_enable_WEBGL_multi_draw","webgl_enable_EXT_polygon_offset_clamp","webgl_enable_EXT_clip_control","webgl_enable_WEBGL_polygon_mode","emscriptenWebGLGet","computeUnpackAlignedImageSize","colorChannelsInGlTextureFormat","emscriptenWebGLGetTexPixelData","emscriptenWebGLGetUniform","webglGetUniformLocation","webglPrepareUniformLocationsBeforeFirstUse","webglGetLeftBracePos","emscriptenWebGLGetVertexAttrib","__glGetActiveAttribOrUniform","writeGLArray","registerWebGlEventCallback","runAndAbortIfError","ALLOC_NORMAL","ALLOC_STACK","allocate","writeStringToMemory","writeAsciiToMemory","demangle","stackTrace","getFunctionArgsName","createJsInvokerSignature","PureVirtualError","registerInheritedInstance","unregisterInheritedInstance","getInheritedInstanceCount","getLiveInheritedInstances","setDelayFunction","count_emval_handles"];sh.forEach(Me);var ah=["run","out","err","callMain","abort","wasmMemory","wasmExports","HEAPF32","HEAPF64","HEAP8","HEAPU8","HEAP16","HEAPU16","HEAP32","HEAPU32","HEAP64","HEAPU64","writeStackCookie","checkStackCookie","INT53_MAX","INT53_MIN","bigintToI53Checked","stackSave","stackRestore","stackAlloc","setTempRet0","ptrToString","exitJS","getHeapMax","growMemory","ENV","ERRNO_CODES","strError","DNS","Protocols","Sockets","timers","warnOnce","readEmAsmArgsArray","readEmAsmArgs","runEmAsmFunction","getExecutableName","keepRuntimeAlive","asyncLoad","alignMemory","mmapAlloc","wasmTable","getUniqueRunDependency","noExitRuntime","addOnPreRun","addOnPostRun","freeTableIndexes","functionsInTableMap","setValue","getValue","PATH","PATH_FS","UTF8Decoder","UTF8ArrayToString","UTF8ToString","stringToUTF8Array","stringToUTF8","lengthBytesUTF8","intArrayFromString","AsciiToString","UTF16Decoder","UTF16ToString","stringToUTF16","lengthBytesUTF16","UTF32ToString","stringToUTF32","lengthBytesUTF32","stringToUTF8OnStack","writeArrayToMemory","JSEvents","specialHTMLTargets","findCanvasEventTarget","currentFullscreenStrategy","restoreOldWindowedStyle","UNWIND_CACHE","ExitStatus","getEnvStrings","checkWasiClock","doReadv","doWritev","initRandomFill","randomFill","emSetImmediate","emClearImmediate_deps","emClearImmediate","promiseMap","uncaughtExceptionCount","exceptionLast","exceptionCaught","ExceptionInfo","findMatchingCatch","getExceptionMessageCommon","Browser","requestFullscreen","requestFullScreen","setCanvasSize","getUserMedia","createContext","getPreloadedImageData__data","wget","MONTH_DAYS_REGULAR","MONTH_DAYS_LEAP","MONTH_DAYS_REGULAR_CUMULATIVE","MONTH_DAYS_LEAP_CUMULATIVE","isLeapYear","ydayFromDate","SYSCALLS","preloadPlugins","FS_modeStringToFlags","FS_getMode","FS_stdin_getChar_buffer","FS_stdin_getChar","FS_readFile","FS_root","FS_mounts","FS_devices","FS_streams","FS_nextInode","FS_nameTable","FS_currentPath","FS_initialized","FS_ignorePermissions","FS_filesystems","FS_syncFSRequests","FS_readFiles","FS_lookupPath","FS_getPath","FS_hashName","FS_hashAddNode","FS_hashRemoveNode","FS_lookupNode","FS_createNode","FS_destroyNode","FS_isRoot","FS_isMountpoint","FS_isFile","FS_isDir","FS_isLink","FS_isChrdev","FS_isBlkdev","FS_isFIFO","FS_isSocket","FS_flagsToPermissionString","FS_nodePermissions","FS_mayLookup","FS_mayCreate","FS_mayDelete","FS_mayOpen","FS_checkOpExists","FS_nextfd","FS_getStreamChecked","FS_getStream","FS_createStream","FS_closeStream","FS_dupStream","FS_doSetAttr","FS_chrdev_stream_ops","FS_major","FS_minor","FS_makedev","FS_registerDevice","FS_getDevice","FS_getMounts","FS_syncfs","FS_mount","FS_unmount","FS_lookup","FS_mknod","FS_statfs","FS_statfsStream","FS_statfsNode","FS_create","FS_mkdir","FS_mkdev","FS_symlink","FS_rename","FS_rmdir","FS_readdir","FS_readlink","FS_stat","FS_fstat","FS_lstat","FS_doChmod","FS_chmod","FS_lchmod","FS_fchmod","FS_doChown","FS_chown","FS_lchown","FS_fchown","FS_doTruncate","FS_truncate","FS_ftruncate","FS_utime","FS_open","FS_close","FS_isClosed","FS_llseek","FS_read","FS_write","FS_mmap","FS_msync","FS_ioctl","FS_writeFile","FS_cwd","FS_chdir","FS_createDefaultDirectories","FS_createDefaultDevices","FS_createSpecialDirectories","FS_createStandardStreams","FS_staticInit","FS_init","FS_quit","FS_findObject","FS_analyzePath","FS_createFile","FS_forceLoadFile","FS_absolutePath","FS_createFolder","FS_createLink","FS_joinPath","FS_mmapAlloc","FS_standardizePath","TTY","PIPEFS","SOCKFS","tempFixedLengthArray","miniTempWebGLFloatBuffers","miniTempWebGLIntBuffers","GL","AL","GLUT","EGL","GLEW","IDBStore","SDL","SDL_gfx","allocateUTF8","allocateUTF8OnStack","print","printErr","jstoi_s","InternalError","BindingError","throwInternalError","throwBindingError","registeredTypes","awaitingDependencies","typeDependencies","tupleRegistrations","structRegistrations","sharedRegisterType","whenDependentTypesAreResolved","getTypeName","getFunctionName","heap32VectorToArray","requireRegisteredType","usesDestructorStack","checkArgCount","getRequiredArgCount","createJsInvoker","UnboundTypeError","GenericWireTypeSize","EmValType","EmValOptionalType","throwUnboundTypeError","ensureOverloadTable","exposePublicSymbol","replacePublicSymbol","createNamedFunction","embindRepr","registeredInstances","getBasestPointer","getInheritedInstance","registeredPointers","registerType","integerReadValueFromPointer","enumReadValueFromPointer","floatReadValueFromPointer","assertIntegerRange","readPointer","runDestructors","craftInvokerFunction","embind__requireFunction","genericPointerToWireType","constNoSmartPtrRawPointerToWireType","nonConstNoSmartPtrRawPointerToWireType","init_RegisteredPointer","RegisteredPointer","RegisteredPointer_fromWireType","runDestructor","releaseClassHandle","finalizationRegistry","detachFinalizer_deps","detachFinalizer","attachFinalizer","makeClassHandle","init_ClassHandle","ClassHandle","throwInstanceAlreadyDeleted","deletionQueue","flushPendingDeletes","delayFunction","RegisteredClass","shallowCopyInternalPointer","downcastPointer","upcastPointer","validateThis","char_0","char_9","makeLegalFunctionName","emval_freelist","emval_handles","emval_symbols","getStringOrSymbol","Emval","emval_get_global","emval_returnValue","emval_lookupTypes","emval_methodCallers","emval_addMethodCaller"];ah.forEach(we),t.incrementExceptionRefcount=nh,t.decrementExceptionRefcount=ih,t.getExceptionMessage=Jc;function oh(){fe("fetchSettings")}var Qc={658116:()=>{typeof t<"u"&&"mjDISABLESTRING mjENABLESTRING mjFRAMESTRING mjLABELSTRING mjRNDSTRING mjTIMERSTRING mjVISSTRING".split(" ").forEach(function(i){Object.defineProperty(t,i,{get:function(){return t["get_"+i]()},set:function(s){},enumerable:!0,configurable:!0})})}},el=ee("___getTypeName"),Ca=ee("_malloc"),Pa=ee("_fflush"),Vn=ee("_free"),Da=ee("_emscripten_stack_get_end"),tl=ee("_strerror"),_e=ee("_setThrew"),nl=ee("__emscripten_tempret_set"),il=ee("_emscripten_stack_init"),rl=ee("__emscripten_stack_restore"),sl=ee("__emscripten_stack_alloc"),al=ee("_emscripten_stack_get_current"),La=ee("___cxa_decrement_exception_refcount"),Ms=ee("___cxa_increment_exception_refcount"),ol=ee("___get_exception_message"),cl=ee("___cxa_can_catch"),ll=ee("___cxa_get_exception_ptr");function ch(i){el=re("__getTypeName",1),Ca=re("malloc",1),Pa=re("fflush",1),Vn=re("free",1),Da=i.emscripten_stack_get_end,i.emscripten_stack_get_base,tl=re("strerror",1),_e=re("setThrew",2),nl=re("_emscripten_tempret_set",1),il=i.emscripten_stack_init,i.emscripten_stack_get_free,rl=i._emscripten_stack_restore,sl=i._emscripten_stack_alloc,al=i.emscripten_stack_get_current,La=re("__cxa_decrement_exception_refcount",1),Ms=re("__cxa_increment_exception_refcount",1),ol=re("__get_exception_message",3),cl=re("__cxa_can_catch",3),ll=re("__cxa_get_exception_ptr",1)}var ul={__assert_fail:yn,__cxa_begin_catch:ss,__cxa_current_primary_exception:as,__cxa_end_catch:Lr,__cxa_find_matching_catch_2:os,__cxa_find_matching_catch_3:qi,__cxa_find_matching_catch_4:cs,__cxa_rethrow:Fr,__cxa_rethrow_primary_exception:ma,__cxa_throw:_a,__cxa_uncaught_exceptions:xa,__resumeException:ga,__syscall_dup3:on,__syscall_fcntl64:tn,__syscall_fstat64:dn,__syscall_ioctl:Qt,__syscall_lstat64:nn,__syscall_newfstatat:Ir,__syscall_openat:Yn,__syscall_stat64:rf,_abort_js:sf,_embind_register_bigint:of,_embind_register_bool:cf,_embind_register_class:Tf,_embind_register_class_class_function:Cf,_embind_register_class_constructor:Pf,_embind_register_class_function:Df,_embind_register_class_property:Lf,_embind_register_constant:Ff,_embind_register_emval:zc,_embind_register_enum:Uf,_embind_register_enum_value:Nf,_embind_register_float:kf,_embind_register_function:Bf,_embind_register_integer:zf,_embind_register_memory_view:Vf,_embind_register_optional:Hf,_embind_register_std_string:Wf,_embind_register_std_wstring:Zf,_embind_register_user_type:Jf,_embind_register_void:Qf,_emscripten_throw_longjmp:ed,_emval_as:td,_emval_call:nd,_emval_call_method:rd,_emval_decref:wa,_emval_get_global:sd,_emval_get_method_caller:cd,_emval_get_property:ld,_emval_incref:ud,_emval_is_number:fd,_emval_is_string:dd,_emval_new_array:hd,_emval_new_cstring:pd,_emval_run_destructors:md,_emval_take_value:_d,_emval_throw:xd,_localtime_js:Ed,_mktime_js:Md,_tzset_js:Td,clock_time_get:Ad,emscripten_asm_const_int:Pd,emscripten_date_now:jc,emscripten_get_heap_max:Dd,emscripten_get_now:$c,emscripten_resize_heap:Id,environ_get:Nd,environ_sizes_get:Od,exit:zd,fd_close:Vd,fd_read:Hd,fd_seek:Wd,fd_write:$d,invoke_ddd:Up,invoke_dddi:Jh,invoke_dddidi:Qh,invoke_ddidi:Zh,invoke_di:ep,invoke_dii:Gh,invoke_diii:bh,invoke_diiii:Kh,invoke_diiiidd:Yh,invoke_diiiidi:Th,invoke_diiiii:xh,invoke_diiiiii:Dh,invoke_diiiiiii:tp,invoke_diiiiiiiii:Ch,invoke_diiiiiiiiiiii:Ph,invoke_fiii:$p,invoke_i:gh,invoke_id:Pp,invoke_ii:fh,invoke_iid:fp,invoke_iidddd:zp,invoke_iidiii:Bh,invoke_iidiiid:Oh,invoke_iidiiiiidi:zh,invoke_iif:Bp,invoke_iii:lh,invoke_iiid:Vh,invoke_iiididdddddd:kh,invoke_iiidiiiiiiii:Nh,invoke_iiii:ph,invoke_iiiidddiiiii:ip,invoke_iiiii:Sh,invoke_iiiiid:Sp,invoke_iiiiii:_p,invoke_iiiiiii:hp,invoke_iiiiiiii:up,invoke_iiiiiiiidd:bp,invoke_iiiiiiiii:jh,invoke_iiiiiiiiii:pp,invoke_iiiiiiiiiidddiiiiiiiii:Uh,invoke_iiiiiiiiiii:Xp,invoke_iiiiiiiiiiii:jp,invoke_iiiiiiiiiiiii:Cp,invoke_iiij:mp,invoke_iiji:yp,invoke_j:Hp,invoke_ji:Rp,invoke_jiiii:xp,invoke_jij:Ap,invoke_v:hh,invoke_vi:dh,invoke_vid:dp,invoke_viddd:gp,invoke_vidddd:vp,invoke_vidi:qh,invoke_vidiii:Fh,invoke_vii:_h,invoke_viid:Xh,invoke_viiddi:wp,invoke_viiddidi:Tp,invoke_viiddii:np,invoke_viidi:Wh,invoke_viidii:Mh,invoke_viidiii:cp,invoke_viidiiid:ap,invoke_viidiiiii:Ih,invoke_viidiiiiidi:lp,invoke_viidiiiiiiii:Lh,invoke_viii:uh,invoke_viiid:Ah,invoke_viiidd:Mp,invoke_viiidi:Hh,invoke_viiididdddddd:op,invoke_viiidiiiiiiii:sp,invoke_viiii:yh,invoke_viiiiddd:Ep,invoke_viiiidi:Np,invoke_viiiifi:Op,invoke_viiiii:mh,invoke_viiiiid:wh,invoke_viiiiii:vh,invoke_viiiiiii:Eh,invoke_viiiiiiii:$h,invoke_viiiiiiiiii:Fp,invoke_viiiiiiiiiidddiiiiiiiii:rp,invoke_viiiiiiiiiiid:Rh,invoke_viiiiiiiiiiiii:Lp,invoke_viiiiiiiiiiiiiii:Yp,invoke_viiiiiiiiiiiiiiiiii:Ip,invoke_viiiij:Vp,invoke_viij:Gp,invoke_viijii:Wp,invoke_vij:kp,invoke_vijjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj:Dp,llvm_eh_typeid_for:jd},Li=await be();function lh(i,s,c){var u=B();try{return xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function uh(i,s,c,u){var d=B();try{xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function fh(i,s){var c=B();try{return xe(i)(s)}catch(u){if(H(c),!(u instanceof y))throw u;_e(1,0)}}function dh(i,s){var c=B();try{xe(i)(s)}catch(u){if(H(c),!(u instanceof y))throw u;_e(1,0)}}function hh(i){var s=B();try{xe(i)()}catch(c){if(H(s),!(c instanceof y))throw c;_e(1,0)}}function ph(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function mh(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function _h(i,s,c){var u=B();try{xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function xh(i,s,c,u,d,x){var M=B();try{return xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function gh(i){var s=B();try{return xe(i)()}catch(c){if(H(s),!(c instanceof y))throw c;_e(1,0)}}function vh(i,s,c,u,d,x,M){var T=B();try{xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function yh(i,s,c,u,d){var x=B();try{xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function Sh(i,s,c,u,d){var x=B();try{return xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function bh(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function Eh(i,s,c,u,d,x,M,T){var I=B();try{xe(i)(s,c,u,d,x,M,T)}catch(X){if(H(I),!(X instanceof y))throw X;_e(1,0)}}function Mh(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Th(i,s,c,u,d,x,M){var T=B();try{return xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function wh(i,s,c,u,d,x,M){var T=B();try{xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function Ah(i,s,c,u,d){var x=B();try{xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function Rh(i,s,c,u,d,x,M,T,I,X,q,ae,ue){var oe=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue)}catch(pe){if(H(oe),!(pe instanceof y))throw pe;_e(1,0)}}function Ch(i,s,c,u,d,x,M,T,I,X){var q=B();try{return xe(i)(s,c,u,d,x,M,T,I,X)}catch(ae){if(H(q),!(ae instanceof y))throw ae;_e(1,0)}}function Ph(i,s,c,u,d,x,M,T,I,X,q,ae,ue){var oe=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue)}catch(pe){if(H(oe),!(pe instanceof y))throw pe;_e(1,0)}}function Dh(i,s,c,u,d,x,M){var T=B();try{return xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function Lh(i,s,c,u,d,x,M,T,I,X,q,ae){var ue=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae)}catch(oe){if(H(ue),!(oe instanceof y))throw oe;_e(1,0)}}function Fh(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Ih(i,s,c,u,d,x,M,T,I){var X=B();try{xe(i)(s,c,u,d,x,M,T,I)}catch(q){if(H(X),!(q instanceof y))throw q;_e(1,0)}}function Uh(i,s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut,gt,rn,$t){var cn=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut,gt,rn,$t)}catch(Lt){if(H(cn),!(Lt instanceof y))throw Lt;_e(1,0)}}function Nh(i,s,c,u,d,x,M,T,I,X,q,ae){var ue=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae)}catch(oe){if(H(ue),!(oe instanceof y))throw oe;_e(1,0)}}function Oh(i,s,c,u,d,x,M){var T=B();try{return xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function kh(i,s,c,u,d,x,M,T,I,X,q,ae){var ue=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae)}catch(oe){if(H(ue),!(oe instanceof y))throw oe;_e(1,0)}}function Bh(i,s,c,u,d,x){var M=B();try{return xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function zh(i,s,c,u,d,x,M,T,I,X){var q=B();try{return xe(i)(s,c,u,d,x,M,T,I,X)}catch(ae){if(H(q),!(ae instanceof y))throw ae;_e(1,0)}}function Vh(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function Gh(i,s,c){var u=B();try{return xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function Hh(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Wh(i,s,c,u,d){var x=B();try{xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function Xh(i,s,c,u){var d=B();try{xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function $h(i,s,c,u,d,x,M,T,I){var X=B();try{xe(i)(s,c,u,d,x,M,T,I)}catch(q){if(H(X),!(q instanceof y))throw q;_e(1,0)}}function jh(i,s,c,u,d,x,M,T,I){var X=B();try{return xe(i)(s,c,u,d,x,M,T,I)}catch(q){if(H(X),!(q instanceof y))throw q;_e(1,0)}}function Yh(i,s,c,u,d,x,M){var T=B();try{return xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function qh(i,s,c,u){var d=B();try{xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function Kh(i,s,c,u,d){var x=B();try{return xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function Zh(i,s,c,u,d){var x=B();try{return xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function Jh(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function Qh(i,s,c,u,d,x){var M=B();try{return xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function ep(i,s){var c=B();try{return xe(i)(s)}catch(u){if(H(c),!(u instanceof y))throw u;_e(1,0)}}function tp(i,s,c,u,d,x,M,T){var I=B();try{return xe(i)(s,c,u,d,x,M,T)}catch(X){if(H(I),!(X instanceof y))throw X;_e(1,0)}}function np(i,s,c,u,d,x,M){var T=B();try{xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function ip(i,s,c,u,d,x,M,T,I,X,q,ae){var ue=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae)}catch(oe){if(H(ue),!(oe instanceof y))throw oe;_e(1,0)}}function rp(i,s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut,gt,rn,$t,cn){var Lt=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut,gt,rn,$t,cn)}catch(Fi){if(H(Lt),!(Fi instanceof y))throw Fi;_e(1,0)}}function sp(i,s,c,u,d,x,M,T,I,X,q,ae,ue){var oe=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue)}catch(pe){if(H(oe),!(pe instanceof y))throw pe;_e(1,0)}}function ap(i,s,c,u,d,x,M,T){var I=B();try{xe(i)(s,c,u,d,x,M,T)}catch(X){if(H(I),!(X instanceof y))throw X;_e(1,0)}}function op(i,s,c,u,d,x,M,T,I,X,q,ae,ue){var oe=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue)}catch(pe){if(H(oe),!(pe instanceof y))throw pe;_e(1,0)}}function cp(i,s,c,u,d,x,M){var T=B();try{xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function lp(i,s,c,u,d,x,M,T,I,X,q){var ae=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q)}catch(ue){if(H(ae),!(ue instanceof y))throw ue;_e(1,0)}}function up(i,s,c,u,d,x,M,T){var I=B();try{return xe(i)(s,c,u,d,x,M,T)}catch(X){if(H(I),!(X instanceof y))throw X;_e(1,0)}}function fp(i,s,c){var u=B();try{return xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function dp(i,s,c){var u=B();try{xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function hp(i,s,c,u,d,x,M){var T=B();try{return xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function pp(i,s,c,u,d,x,M,T,I,X){var q=B();try{return xe(i)(s,c,u,d,x,M,T,I,X)}catch(ae){if(H(q),!(ae instanceof y))throw ae;_e(1,0)}}function mp(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function _p(i,s,c,u,d,x){var M=B();try{return xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function xp(i,s,c,u,d){var x=B();try{return xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;return _e(1,0),0n}}function gp(i,s,c,u,d){var x=B();try{xe(i)(s,c,u,d)}catch(M){if(H(x),!(M instanceof y))throw M;_e(1,0)}}function vp(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function yp(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function Sp(i,s,c,u,d,x){var M=B();try{return xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function bp(i,s,c,u,d,x,M,T,I,X){var q=B();try{return xe(i)(s,c,u,d,x,M,T,I,X)}catch(ae){if(H(q),!(ae instanceof y))throw ae;_e(1,0)}}function Ep(i,s,c,u,d,x,M,T){var I=B();try{xe(i)(s,c,u,d,x,M,T)}catch(X){if(H(I),!(X instanceof y))throw X;_e(1,0)}}function Mp(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Tp(i,s,c,u,d,x,M,T){var I=B();try{xe(i)(s,c,u,d,x,M,T)}catch(X){if(H(I),!(X instanceof y))throw X;_e(1,0)}}function wp(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Ap(i,s,c){var u=B();try{return xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;return _e(1,0),0n}}function Rp(i,s){var c=B();try{return xe(i)(s)}catch(u){if(H(c),!(u instanceof y))throw u;return _e(1,0),0n}}function Cp(i,s,c,u,d,x,M,T,I,X,q,ae,ue){var oe=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue)}catch(pe){if(H(oe),!(pe instanceof y))throw pe;_e(1,0)}}function Pp(i,s){var c=B();try{return xe(i)(s)}catch(u){if(H(c),!(u instanceof y))throw u;_e(1,0)}}function Dp(i,s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut,gt,rn,$t,cn,Lt,Fi,Jp,Qp,em,tm,nm,im,rm,sm,am,om,cm,lm,um,fm,dm,hm,pm,mm,_m,xm,gm,vm,ym,Sm,bm,Em,Mm,Tm,wm,Am,Rm,Cm,Pm,Dm,Lm,Fm,Im,Um,Nm,Om,km,Bm,zm,Vm,Gm,Hm,Wm,Xm,$m,jm,Ym,qm,Km,Zm,Jm,Qm,e0,t0,n0,i0,r0){var s0=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut,gt,rn,$t,cn,Lt,Fi,Jp,Qp,em,tm,nm,im,rm,sm,am,om,cm,lm,um,fm,dm,hm,pm,mm,_m,xm,gm,vm,ym,Sm,bm,Em,Mm,Tm,wm,Am,Rm,Cm,Pm,Dm,Lm,Fm,Im,Um,Nm,Om,km,Bm,zm,Vm,Gm,Hm,Wm,Xm,$m,jm,Ym,qm,Km,Zm,Jm,Qm,e0,t0,n0,i0,r0)}catch(dl){if(H(s0),!(dl instanceof y))throw dl;_e(1,0)}}function Lp(i,s,c,u,d,x,M,T,I,X,q,ae,ue,oe){var pe=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue,oe)}catch(Xe){if(H(pe),!(Xe instanceof y))throw Xe;_e(1,0)}}function Fp(i,s,c,u,d,x,M,T,I,X,q){var ae=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q)}catch(ue){if(H(ae),!(ue instanceof y))throw ue;_e(1,0)}}function Ip(i,s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut){var gt=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe,ht,rt,Ut)}catch(rn){if(H(gt),!(rn instanceof y))throw rn;_e(1,0)}}function Up(i,s,c){var u=B();try{return xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function Np(i,s,c,u,d,x,M){var T=B();try{xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function Op(i,s,c,u,d,x,M){var T=B();try{xe(i)(s,c,u,d,x,M)}catch(I){if(H(T),!(I instanceof y))throw I;_e(1,0)}}function kp(i,s,c){var u=B();try{xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function Bp(i,s,c){var u=B();try{return xe(i)(s,c)}catch(d){if(H(u),!(d instanceof y))throw d;_e(1,0)}}function zp(i,s,c,u,d,x){var M=B();try{return xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Vp(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Gp(i,s,c,u){var d=B();try{xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function Hp(i){var s=B();try{return xe(i)()}catch(c){if(H(s),!(c instanceof y))throw c;return _e(1,0),0n}}function Wp(i,s,c,u,d,x){var M=B();try{xe(i)(s,c,u,d,x)}catch(T){if(H(M),!(T instanceof y))throw T;_e(1,0)}}function Xp(i,s,c,u,d,x,M,T,I,X,q){var ae=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q)}catch(ue){if(H(ae),!(ue instanceof y))throw ue;_e(1,0)}}function $p(i,s,c,u){var d=B();try{return xe(i)(s,c,u)}catch(x){if(H(d),!(x instanceof y))throw x;_e(1,0)}}function jp(i,s,c,u,d,x,M,T,I,X,q,ae){var ue=B();try{return xe(i)(s,c,u,d,x,M,T,I,X,q,ae)}catch(oe){if(H(ue),!(oe instanceof y))throw oe;_e(1,0)}}function Yp(i,s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe){var ht=B();try{xe(i)(s,c,u,d,x,M,T,I,X,q,ae,ue,oe,pe,Xe)}catch(rt){if(H(ht),!(rt instanceof y))throw rt;_e(1,0)}}var fl;function qp(){il(),Z()}function Fa(){if(Ie>0){Rt=Fa;return}if(qp(),ct(),Ie>0){Rt=Fa;return}function i(){A(!fl),fl=!0,t.calledRun=!0,!z&&(lt(),Ge?.(t),t.onRuntimeInitialized?.(),J("onRuntimeInitialized"),A(!t._main,'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]'),Dt())}t.setStatus?(t.setStatus("Running..."),setTimeout(()=>{setTimeout(()=>t.setStatus(""),1),i()},1)):i(),D()}function Kp(){var i=U,s=L,c=!1;U=L=u=>{c=!0};try{Pa(0),["stdout","stderr"].forEach(u=>{var d=S.analyzePath("/dev/"+u);if(d){var x=d.object,M=x.rdev,T=Fe.ttys[M];T?.output?.length&&(c=!0)}})}catch{}U=i,L=s,c&&Le("stdio streams had content in them that was not flushed. you should set EXIT_RUNTIME to 1 (see the Emscripten FAQ), or make sure to emit a newline when you printf etc.")}function Zp(){if(t.preInit)for(typeof t.preInit=="function"&&(t.preInit=[t.preInit]);t.preInit.length>0;)t.preInit.shift()();J("preInit")}Zp(),Fa(),Mt?e=t:e=new Promise((i,s)=>{Ge=i,ot=s});for(const i of Object.keys(t))i in r||Object.defineProperty(r,i,{configurable:!0,get(){$(`Access to module property ('${i}') is no longer possible via the module constructor argument; Instead, use the result of the module constructor.`)}});return e});const GS=""+new URL("mujoco-Bp43jdDU.wasm",import.meta.url).href,Qe=r=>document.getElementById(r),Ar=new URL(location.href).searchParams,Ti=Ar.get("embed")==="1";let yc=Ar.get("autoplay")==="1",pa=Ar.get("loop")==="1"&&Ar.get("outcome")==="SUCCESS",yr=Ar.get("outcome")||"",Sc=Ar.get("reason")||"";Ti&&document.body.classList.add("embedded");const bc=Qe("scene"),Xn=new b_;Xn.background=new xt(1187627);const Mi=new Tn(50,1,.02,500);Mi.up.set(0,0,1);Mi.position.set(5,-5,4);const rs=new ES({antialias:!0,powerPreference:"high-performance"});rs.setPixelRatio(Math.min(window.devicePixelRatio,2));bc.appendChild(rs.domElement);const ci=new TS(Mi,rs.domElement);ci.enableDamping=!0;ci.minDistance=.25;ci.maxDistance=100;const HS=new I_(16777215,1.8);Xn.add(HS);const ef=new F_(16777215,2.6);ef.position.set(4,2,10);Xn.add(ef);let Ln=null,At=null,En=null,St=[],ca=[],Un=null,ri=0,si=!1,na=!0,Ec=[],Mc=[],dr=!Ti,Jr=[],mn=null,Qr=[],hr=!Ti,pr=!0,mr=Ti,_r="aircraft",la=performance.now(),Fn=0,qs=null,ho=!1,gu=0;function gi(r,e,t={}){Ti&&parent!==window&&parent.postMessage({type:"dp-wasm-progress",record:e?.record||"",stage:r,...t},location.origin)}function zi(r){Qe("status").textContent=r}function ic(){si=!1,Qe("play").textContent="Play",Fn=0,Qe("end-banner").hidden=!0;for(const r of ca)Xn.remove(r),r.geometry?.dispose(),r.material?.dispose();ca=[],Ec=[],Mc=[],Jr=[],Qr=[],Un&&(Xn.remove(Un),Un.geometry.dispose(),Un.material.dispose(),Un=null),mn&&(Xn.remove(mn),mn.geometry.dispose(),mn.material.dispose(),mn=null),En?.delete(),At?.delete(),At=null,En=null,St=[]}function WS(r,e,t){const n=Ln.mjtGeom;if(r===n.mjGEOM_PLANE.value)return new is(Math.max(1,e[0]*2),Math.max(1,e[1]*2));if(r===n.mjGEOM_SPHERE.value)return new $r(e[0],14,10);if(r===n.mjGEOM_CAPSULE.value){const a=new xc(e[0],2*e[1],8,10);return a.rotateX(Math.PI/2),a}if(r===n.mjGEOM_ELLIPSOID.value){const a=new $r(1,14,10);return a.scale(...e),a}if(r===n.mjGEOM_CYLINDER.value){const a=new gc(e[0],e[0],2*e[1],16);return a.rotateX(Math.PI/2),a}if(r===n.mjGEOM_BOX.value)return new Pr(...e.map(a=>2*a));if(r===n.mjGEOM_MESH.value){const a=At.geom_dataid[t],o=At.mesh_vertadr[a],l=At.mesh_vertnum[a],f=At.mesh_faceadr[a],p=At.mesh_facenum[a],h=Float32Array.from(At.mesh_vert.subarray(o*3,(o+l)*3)),_=Array.from(At.mesh_face.subarray(f*3,(f+p)*3)),g=new fn;return g.setAttribute("position",new An(h,3)),g.setIndex(_),g.computeVertexNormals(),g}return new $r(Math.max(e[0],.02))}function XS(r){const e=Array.from(At.geom_rgba.subarray(r*4,r*4+4)),t=At.geom_matid[r];if(t>=0&&e[3]!==0){const n=Array.from(At.mat_rgba.subarray(t*4,t*4+4));n[3]>0&&e.splice(0,4,...n)}return new R_({color:new xt(e[0],e[1],e[2]),opacity:e[3],transparent:e[3]<1,side:Hn,roughness:.75,metalness:.04})}function tf(){for(const r of Mc){const e=r.material;e.transparent=mr||r.userData.originalTransparent,e.opacity=mr?Math.min(.25,r.userData.originalOpacity):r.userData.originalOpacity,e.depthWrite=!mr,e.needsUpdate=!0}Qe("obstacles").textContent=`Obstacles: ${mr?"Translucent":"Solid"}`}function $S(){let r=0;const e=At.names,t=At.name_geomadr,n=new TextDecoder;for(let l=0;l<At.ngeom;l++){if(At.geom_rgba[l*4+3]<.015||At.geom_group[l]>=4)continue;let f="";if(e&&t){const _=t[l];let g=_;for(;g<e.length&&e[g]!==0;)g++;if(f=n.decode(e.subarray(_,g)),f.startsWith("replay_actual_"))continue}const p=Array.from(At.geom_size.subarray(l*3,l*3+3)),h=new li(WS(At.geom_type[l],p,l),XS(l));h.userData.geomId=l,f.startsWith("replay_hit_")&&(h.visible=dr,Ec.push(h)),f.startsWith("replay_map_")&&(h.visible=hr,Jr.push(h)),At.geom_group[l]<2&&!f.startsWith("replay_")&&(h.userData.originalOpacity=h.material.opacity,h.userData.originalTransparent=h.material.transparent,Mc.push(h)),At.geom_group[l]===2&&At.geom_type[l]===Ln.mjtGeom.mjGEOM_MESH.value&&r++,Xn.add(h),ca.push(h)}const a=St.map(l=>l.qpos.slice(0,3)),o=new fn().setFromPoints(a.map(l=>new K(...l)));return Un=new Hu(o,new _c({color:4836853,opacity:.85,transparent:!0})),Xn.add(Un),tf(),r}function jS(r){const e=r?.sfc_pairs||[];if(!Array.isArray(e)||!e.every(a=>Array.isArray(a)&&a.length===2&&a.every(Number.isInteger)))throw new Error("Malformed SFC replay slots");if(e.some(a=>a.some(o=>o<0||o>=At.nmocap)))throw new Error("SFC replay slots exceed model mocap count");if(Qr=e,!e.length)return;const t=new Float32Array(e.length*6),n=new fn;n.setAttribute("position",new An(t,3).setUsage(j0)),n.setDrawRange(0,0),mn=new A_(n,new _c({color:16762460,transparent:!0,opacity:.92,depthWrite:!1})),mn.frustumCulled=!1,mn.visible=pr,Xn.add(mn)}function YS(r){if(!mn)return;const e=mn.geometry.attributes.position;let t=0;for(const[n,a]of Qr){const o=r.mocap_pos[n],l=r.mocap_pos[a];if(!(!o||!l||o[2]<-1e4||l[2]<-1e4)){for(let f=0;f<3;f++)e.array[t*6+f]=o[f],e.array[t*6+3+f]=l[f];t++}}mn.geometry.setDrawRange(0,t*2),e.needsUpdate=!0}function vu(r){let e=0,t=St.length-1;for(;e<t;){const n=e+t>>1;St[n].time<r?e=n+1:t=n}return Math.abs(St[e].time-r)<Math.abs(St[Math.max(0,e-1)].time-r)?e:Math.max(0,e-1)}function Sr(r){if(!At)return;ri=Math.max(0,Math.min(St.length-1,r));const e=St[ri];En.time=e.time,En.qpos.set(e.qpos),En.mocap_pos.set(e.mocap_pos.flat()),En.mocap_quat.set(e.mocap_quat.flat()),Ln.mj_forward(At,En),YS(e);const t=new Bt;for(const a of ca){const o=a.userData.geomId;a.position.set(En.geom_xpos[3*o],En.geom_xpos[3*o+1],En.geom_xpos[3*o+2]);const l=En.geom_xmat.subarray(9*o,9*o+9);t.set(l[0],l[1],l[2],0,l[3],l[4],l[5],0,l[6],l[7],l[8],0,0,0,0,1),a.quaternion.setFromRotationMatrix(t)}if(na){const a=ci.target.clone();ci.target.set(...e.qpos.slice(0,3)),Mi.position.add(ci.target.clone().sub(a))}Qe("timeline").value=String(ri),Qe("time").textContent=`${e.time.toFixed(3)} / ${St.at(-1).time.toFixed(3)} s`;const n=Qe("end-banner");ri===St.length-1&&yr&&!pa?(n.textContent=yr==="METHOD_FAILURE"?`Flight stopped at ${e.time.toFixed(2)} s: ${Sc||"METHOD_FAILURE"}. Recording ends at the true failure time.`:`Flight ended at ${e.time.toFixed(2)} s: ${yr}. This is the recorded physical outcome.`,n.hidden=!1):n.hidden=!0}function nf(){if(!St.length)return;const r=St[ri].qpos.slice(0,3);ci.target.set(...r);const e=Ti&&At.nmocap>100?[.85,-1.12,.62]:[.4,-.55,.32],t=_r==="aircraft"?e:_r==="course"?[3.2,-3.2,2.1]:[1.6,-2.1,8];Mi.position.set(r[0]+t[0],r[1]+t[1],r[2]+t[2]),ci.update()}async function qS(r){const e=Qe("record").value.trim();if(!e)return zi("Specify a browser_replay.json URL");Qe("load").disabled=!0,ic(),zi("Loading recorded poses and MJCF...");const t=performance.now();gi("fetch",r);try{const n=new URL(e,location.href);if(n.origin!==location.origin)throw new Error("Record must be served by the same SSH-forwarded HTTP server");const a=await fetch(n);if(!a.ok)throw new Error(`HTTP ${a.status} loading ${n.pathname}`);const o=await a.json();if(o.format!=="drone-playground-mujoco-replay-v1"||!Array.isArray(o.frames)||o.frames.length<2)throw new Error("Incompatible recorded MuJoCo episode");Ln||(zi("Initializing MuJoCo WebAssembly..."),gi("wasm",r),Ln=await VS({locateFile:p=>p.endsWith(".wasm")?GS:p}),Ln.FS.mkdir("/working"),Ln.FS.mount(Ln.MEMFS,{root:"."},"/working")),gi("assets",r);for(const p of[o.model,...o.assets]){if(!/^[A-Za-z0-9_.-]+$/.test(p))throw new Error(`Unsupported model filename: ${p}`);const h=await fetch(new URL(p,n));if(!h.ok)throw new Error(`Missing MJCF asset ${p}`);const _=new Uint8Array(await h.arrayBuffer());Ln.FS.writeFile(`/working/${p}`,_)}zi("Preparing native 3D scene..."),gi("scene",r),await new Promise(p=>requestAnimationFrame(()=>setTimeout(p,0))),At=Ln.MjModel.mj_loadXML(`/working/${o.model}`),En=new Ln.MjData(At),gi("geometry",r),St=o.frames;for(const p of St)if(p.qpos.length!==At.nq||p.mocap_pos.length!==At.nmocap)throw new Error("Recorded MuJoCo dimensions differ from loaded MJCF");const l=$S();jS(o.overlays),Qe("hits").textContent=`Sensor hits: ${dr?"On":"Off"}`,Qe("map").textContent=`MAP: ${hr?"On":"Off"}`,Qe("map").disabled=Jr.length===0,Qe("sfc").textContent=`SFC: ${pr?"On":"Off"}`,Qe("sfc").disabled=Qr.length===0,Sr(0),nf(),Fn=St[0].time,Qe("timeline").max=String(St.length-1),Qe("timeline").disabled=!1,Qe("play").disabled=!1,zi(`${St.length} frames · ${l} Crazyflie mesh geoms · MAP ${Jr.length} sampled voxels · SFC ${Qr.length} edges`);const f=new URL(location.href);return f.searchParams.set("record",e),history.replaceState(null,"",f),yc&&(si=!0,Qe("play").textContent="Pause",la=performance.now()),gi("ready",r,{frames:St.length,duration:St.at(-1).time,elapsed_ms:Math.round(performance.now()-t)}),!0}catch(n){return zi(`Cannot load recording: ${n.message}`),console.error(n),ic(),gi("error",r,{message:n.message}),!1}finally{Qe("load").disabled=!1}}function es(r){qs=r,!ho&&(ho=!0,(async()=>{try{for(;qs;){const e=qs;qs=null,Qe("record").value=String(e.record||""),yc=e.autoplay===!0,yr=String(e.outcome||""),Sc=String(e.reason||""),pa=e.loop===!0&&yr==="SUCCESS",e.record?await qS(e):(ic(),zi("Ready"),gi("cleared",e))}}finally{ho=!1}})())}Qe("load").addEventListener("click",()=>es({record:Qe("record").value.trim()}));Qe("record").addEventListener("keydown",r=>{r.key==="Enter"&&es({record:Qe("record").value.trim()})});Qe("play").addEventListener("click",()=>{St.length&&(!si&&ri===St.length-1&&(Fn=St[0].time,Sr(0)),si=!si,Qe("play").textContent=si?"Pause":"Play",la=performance.now())});Qe("timeline").addEventListener("input",r=>{si=!1,Qe("play").textContent="Play",Sr(Number(r.target.value)),Fn=St[ri].time});Qe("follow").addEventListener("click",()=>{na=!na,Qe("follow").textContent=`Follow drone: ${na?"On":"Off"}`});Qe("view").addEventListener("click",()=>{const r=["aircraft","course","overhead"];_r=r[(r.indexOf(_r)+1)%r.length],Qe("view").textContent=`Camera: ${_r[0].toUpperCase()+_r.slice(1)}`,nf()});Qe("trail").addEventListener("click",()=>{Un&&(Un.visible=!Un.visible,Qe("trail").textContent=`Flight path: ${Un.visible?"On":"Off"}`)});Qe("hits").addEventListener("click",()=>{dr=!dr;for(const r of Ec)r.visible=dr;Qe("hits").textContent=`Sensor hits: ${dr?"On":"Off"}`});Qe("map").addEventListener("click",()=>{hr=!hr;for(const r of Jr)r.visible=hr;Qe("map").textContent=`MAP: ${hr?"On":"Off"}`});Qe("sfc").addEventListener("click",()=>{pr=!pr,mn&&(mn.visible=pr),Qe("sfc").textContent=`SFC: ${pr?"On":"Off"}`});Qe("obstacles").addEventListener("click",()=>{mr=!mr,tf()});window.addEventListener("resize",Tc);function Tc(){const{width:r,height:e}=bc.getBoundingClientRect();Mi.aspect=r/Math.max(e,1),Mi.updateProjectionMatrix(),rs.setSize(r,e)}new ResizeObserver(Tc).observe(bc);function rc(r){if(Ti&&r-gu<1e3/30){requestAnimationFrame(rc);return}if(gu=r,si&&St.length){const e=Math.min(.3,Math.max(0,(r-la)/1e3))*Number(Qe("speed").value);if(Fn+=e,Fn>=St.at(-1).time)if(pa){const t=St.at(-1).time-St[0].time;Fn=t>0?St[0].time+(Fn-St[0].time)%t:St[0].time;const n=vu(Fn);n!==ri&&Sr(n)}else Fn=St.at(-1).time,Sr(St.length-1),si=!1,Qe("play").textContent="Play";else{const t=vu(Fn);t!==ri&&Sr(t)}}la=r,ci.update(),rs.render(Xn,Mi),requestAnimationFrame(rc)}Tc();requestAnimationFrame(rc);const yu=new URL(location.href).searchParams.get("record");yu&&es({record:yu,autoplay:yc,outcome:yr,reason:Sc,loop:pa});Ti&&parent!==window&&(window.addEventListener("message",r=>{r.origin!==location.origin||r.source!==window.parent||(r.data?.type==="dp-wasm-load"?es({record:r.data.record,autoplay:r.data.autoplay,outcome:r.data.outcome,reason:r.data.reason,loop:r.data.loop}):r.data?.type==="dp-wasm-clear"&&es({record:""}))}),parent.postMessage({type:"dp-wasm-ready"},location.origin));const KS=Object.freeze(Object.defineProperty({__proto__:null},Symbol.toStringTag,{value:"Module"}));
