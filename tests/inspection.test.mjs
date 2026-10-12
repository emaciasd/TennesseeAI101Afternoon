import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classify, modelBase, csv } from '../src/inspection.mjs';
const predictions = [{className:'Correct',probability:.97},{className:'Incorrect',probability:.03}];
test('only the mapped confident winner can pass',()=>{
 assert.equal(classify(predictions,'Correct','Incorrect',.9).verdict,'GO');
 assert.equal(classify(predictions,'Incorrect','Correct',.9).verdict,'NO-GO');
 assert.equal(classify(predictions,'Correct','Incorrect',.99).verdict,'REVIEW');
 assert.equal(classify([{className:'Empty',probability:.99},...predictions],'Correct','Incorrect',.9).verdict,'REVIEW');
 assert.equal(classify(predictions,'Correct','Correct',.9).verdict,'REVIEW');
 assert.equal(classify([],'Correct','Incorrect',.9).verdict,'REVIEW');
});
test('normalizes model URLs and rejects insecure remote endpoints',()=>{
 assert.equal(modelBase('https://teachablemachine.withgoogle.com/models/abc/model.json'),'https://teachablemachine.withgoogle.com/models/abc/');
 assert.equal(modelBase('http://localhost:5173/test-model'),'http://localhost:5173/test-model/');
 assert.throws(()=>modelBase('http://example.com/model/'));
 assert.throws(()=>modelBase('javascript:alert(1)'));
});
test('CSV preserves inspection data and escapes spreadsheet formulas',()=>{
 const output=csv([{time:0,verdict:'GO',confidence:.97,label:'=SUM(1,2)',modelUrl:'https://example.com/'}]);
 assert.match(output,/1970-01-01T00:00:00.000Z/);
 assert.match(output,/97.00/);
 assert.match(output,/"'=SUM\(1,2\)"/);
});
